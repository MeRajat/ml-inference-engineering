#!/usr/bin/env python3
"""Check that every http(s) link in the repository's Markdown files resolves.

Usage: python3 scripts/check_links.py [--timeout 20]

Exits non-zero if any link fails. Sites that block automated clients (HTTP 403
or 429) are reported as "blocked" and do not fail the run: open those by hand.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import re
import sys
import urllib.error
import urllib.request

LINK_RE = re.compile(r"\]\((https?://[^)\s]+)\)|<(https?://[^>\s]+)>")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
BLOCKED = {401, 403, 405, 406, 418, 429, 999}


def find_links(root: pathlib.Path) -> dict[str, list[str]]:
    links: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.md")):
        if ".git" in path.parts:
            continue
        for match in LINK_RE.finditer(path.read_text(encoding="utf-8")):
            url = (match.group(1) or match.group(2)).rstrip(".,")
            links.setdefault(url, []).append(str(path.relative_to(root)))
    return links


def check(url: str, timeout: float) -> tuple[str, str]:
    target = url.split("#", 1)[0]
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(
            target, method=method, headers={"User-Agent": USER_AGENT, "Accept": "*/*"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return "ok", str(response.status)
        except urllib.error.HTTPError as error:
            if method == "HEAD":
                continue
            if error.code in BLOCKED:
                return "blocked", str(error.code)
            return "fail", str(error.code)
        except Exception as error:  # noqa: BLE001 - report any network failure
            if method == "HEAD":
                continue
            return "fail", type(error).__name__
    return "fail", "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    links = find_links(root)
    counts = {"ok": 0, "blocked": 0, "fail": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(check, url, args.timeout): url for url in links}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            status, detail = future.result()
            counts[status] += 1
            if status != "ok":
                print(f"{status.upper():8} {detail:>16}  {url}  ({', '.join(sorted(set(links[url])))})")

    print(f"\n{len(links)} links: {counts['ok']} ok, {counts['blocked']} blocked, {counts['fail']} failed")
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
