#!/usr/bin/env python3
"""
crawl_js_async.py — Async crawl to extract all JavaScript file URLs.

Examples:
  # Same-site only
  python crawl_js_async.py https://example.com --max-pages 800 --max-depth 4

  # Include external CDNs, more concurrency
  python crawl_js_async.py https://example.com --include-external true --concurrency 20

Notes:
- Finds JS via <script src>, <link rel=modulepreload>, .js/.mjs/.cjs in links,
  and common dynamic import/require patterns in HTML.
- Saves a deduped list (one URL per line) to js_<domain>.txt by default.
"""

import argparse
import asyncio
import re
import sys
from collections import deque
from urllib.parse import urljoin, urldefrag, urlparse

import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm
import urllib.robotparser as robotparser

DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

JS_EXTS = (".js", ".mjs", ".cjs")
JS_HINT_RE = re.compile(
    r"""
    (?:                              # common dynamic loaders:
       import\s*\(\s*['"]([^'"]+)['"]\s*\)   # import('...js')
      |require\s*\(\s*['"]([^'"]+)['"]\s*\)  # require('...js')
      |data-main\s*=\s*['"]([^'"]+)['"]      # requirejs data-main
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

def norm_url(base, maybe_rel):
    if not maybe_rel:
        return None
    absu = urljoin(base, maybe_rel)
    absu = urldefrag(absu)[0]
    return absu

def same_site(a, b):
    pa, pb = urlparse(a), urlparse(b)
    return (pa.scheme, pa.hostname, pa.port) == (pb.scheme, pb.hostname, pb.port)

def looks_like_js_url(u: str) -> bool:
    if not u:
        return False
    ul = u.lower()
    if any(ul.endswith(ext) for ext in JS_EXTS):
        return True
    if "format=js" in ul or "module=1" in ul:
        return True
    return False

def is_html_content_type(headers: aiohttp.typedefs.LooseHeaders) -> bool:
    ctype = (headers.get("Content-Type") or "").lower()
    return "text/html" in ctype or "application/xhtml" in ctype

def is_js_content_type(headers: aiohttp.typedefs.LooseHeaders) -> bool:
    ctype = (headers.get("Content-Type") or "").lower()
    return ("javascript" in ctype) or ctype.endswith("/ecmascript")

async def load_robots(session: aiohttp.ClientSession, root: str):
    """Fetch robots.txt and return a configured RobotFileParser (or None on failure)."""
    rp = robotparser.RobotFileParser()
    robots_url = urljoin(root if root.endswith("/") else root + "/", "robots.txt")
    try:
        async with session.get(robots_url, timeout=15) as resp:
            if resp.status == 200:
                text = await resp.text(errors="ignore")
                rp.parse(text.splitlines())
                return rp
    except Exception:
        return None
    return None

async def head_or_get_is_js(session: aiohttp.ClientSession, url: str) -> bool:
    try:
        async with session.head(url, allow_redirects=True, timeout=12) as r:
            if r.status < 400 and is_js_content_type(r.headers):
                return True
    except Exception:
        pass
    # Fallback: quick GET (no body read)
    try:
        async with session.get(url, allow_redirects=True, timeout=15) as r:
            if r.status < 400 and is_js_content_type(r.headers):
                return True
    except Exception:
        pass
    return False

def extract_links_and_js(base_url: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    page_links, js_links = set(), set()

    # A tags (for crawl frontier)
    for a in soup.find_all("a", href=True):
        u = norm_url(base_url, a["href"])
        if u:
            page_links.add(u)

    # Script tags
    for s in soup.find_all("script", src=True):
        u = norm_url(base_url, s["src"])
        if u:
            js_links.add(u)

    # Modulepreload & preloads that point to .js/.mjs
    for l in soup.find_all("link", href=True):
        rel = (l.get("rel") or [])
        u = norm_url(base_url, l["href"])
        if not u:
            continue
        if "modulepreload" in [r.lower() for r in rel]:
            js_links.add(u)
        elif looks_like_js_url(u):
            js_links.add(u)

    # Dynamic patterns inside HTML
    for m in JS_HINT_RE.finditer(html):
        candidate = next((g for g in m.groups() if g), None)
        if candidate:
            u = norm_url(base_url, candidate)
            if u and looks_like_js_url(u):
                js_links.add(u)

    return page_links, js_links

async def crawl(start_url: str,
                max_pages: int,
                max_depth: int,
                include_external: bool,
                concurrency: int,
                honor_robots: bool,
                out_path: str):

    parsed = urlparse(urldefrag(start_url)[0])
    start_root = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
        start_root += f":{parsed.port}"

    # Prepare session
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=20)
    headers = {"User-Agent": DEFAULT_UA, "Accept": "*/*"}
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, ssl=False)

    visited_html = set()
    found_js = set()
    queue = deque([(start_url, 0)])

    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(timeout=timeout, headers=headers, connector=connector) as session:
        robots = None
        if honor_robots:
            robots = await load_robots(session, start_root)

        async def allowed(u: str) -> bool:
            if not honor_robots or robots is None:
                return True
            try:
                return robots.can_fetch(headers["User-Agent"], u)
            except Exception:
                return True

        async def fetch_html(u: str):
            async with sem:
                if not await allowed(u):
                    return None, None
                try:
                    async with session.get(u, allow_redirects=True) as r:
                        if r.status >= 400:
                            return None, None
                        if is_html_content_type(r.headers):
                            text = await r.text(errors="ignore")
                            return str(r.url), text
                        # If the user fed a JS URL directly, collect it
                        if is_js_content_type(r.headers) or looks_like_js_url(str(r.url)):
                            found_js.add(str(r.url))
                            return None, None
                        return None, None
                except Exception:
                    return None, None

        pbar = tqdm(total=max_pages, desc="Crawling pages", unit="page")
        processed = 0

        while queue and processed < max_pages:
            u, depth = queue.popleft()
            if u in visited_html:
                continue

            # Stay on site unless external allowed
            if not include_external and not same_site(start_url, u):
                continue

            final_url, html = await fetch_html(u)
            visited_html.add(u)
            processed += 1
            pbar.update(1)
            pbar.set_postfix({"js": len(found_js)})

            if not html or final_url is None:
                continue

            page_links, js_links = extract_links_and_js(final_url, html)

            # Accumulate JS links (respect include_external flag)
            for j in js_links:
                if include_external or same_site(start_url, j):
                    found_js.add(j)

            # Expand crawl frontier
            if depth < max_depth:
                for nxt in page_links:
                    if (include_external or same_site(start_url, nxt)) and (nxt not in visited_html):
                        queue.append((nxt, depth + 1))

        pbar.close()

        # Verify ambiguous JS (those without .js/.mjs/.cjs)
        # Keep .js/.mjs/.cjs as-is; check others by HEAD/GET content-type
        to_verify = [u for u in found_js if not looks_like_js_url(u)]
        if to_verify:
            vbar = tqdm(total=len(to_verify), desc="Verifying JS", unit="file")
            verified = set()
            async def verify_one(url):
                ok = await head_or_get_is_js(session, url)
                vbar.update(1)
                return url if ok else None

            tasks = [asyncio.create_task(verify_one(u)) for u in to_verify]
            for t in asyncio.as_completed(tasks):
                res = await t
                if res:
                    verified.add(res)
            vbar.close()
            # Merge: keep all sure-by-extension + verified ambiguous
            found = {u for u in found_js if looks_like_js_url(u)}
            found.update(verified)
        else:
            found = set(found_js)

    results = sorted(found)

    if not out_path:
        out_path = f"js_{parsed.hostname or 'results'}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for u in results:
            f.write(u + "\n")

    print(f"✅ Done. Found {len(results)} JavaScript file(s). Saved to: {out_path}")


def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}

def main():
    ap = argparse.ArgumentParser(description="Async website crawler that lists JS file URLs it discovers.")
    ap.add_argument("url", help="Starting URL (e.g., https://example.com)")
    ap.add_argument("--max-pages", type=int, default=500, help="Max pages to fetch (default: 500)")
    ap.add_argument("--max-depth", type=int, default=4, help="Max crawl depth (default: 4)")
    ap.add_argument("--include-external", type=parse_bool, default=False, help="Include external domains (default: false)")
    ap.add_argument("--concurrency", type=int, default=12, help="Concurrent fetches (default: 12)")
    ap.add_argument("--no-robots", action="store_true", help="Ignore robots.txt (not recommended)")
    ap.add_argument("--out", default="", help="Output file path (default: js_<domain>.txt)")
    args = ap.parse_args()

    try:
        asyncio.run(
            crawl(
                start_url=args.url,
                max_pages=max(1, args.max_pages),
                max_depth=max(0, args.max_depth),
                include_external=args.include_externals if hasattr(args, "include_externals") else bool(args.include_external),
                concurrency=max(1, args.concurrency),
                honor_robots=not args.no_robots,
                out_path=args.out.strip(),
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)

if __name__ == "__main__":
    main()
