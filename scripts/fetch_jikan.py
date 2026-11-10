"""Download anime metadata from the Jikan API (unofficial MyAnimeList API).

Run this on your own machine to build the full dataset:

    python scripts/fetch_jikan.py --pages 40 --out data/anime.csv

Each page holds 25 titles, so 40 pages is about 1,000 of the top-ranked anime.
Jikan allows roughly 3 requests/second and 60/minute, so the script sleeps
between calls. The app automatically prefers data/anime.csv over the sample.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests

API = "https://api.jikan.moe/v4/top/anime"
FIELDS = ["title", "genres", "themes", "synopsis", "type", "episodes", "score", "year", "studio", "members"]


def names(items: list[dict] | None) -> str:
    return ", ".join(i["name"] for i in (items or []))


def to_row(a: dict) -> dict:
    year = a.get("year") or ((a.get("aired") or {}).get("prop", {}).get("from", {}) or {}).get("year")
    return {
        "title": a.get("title_english") or a.get("title"),
        "genres": names(a.get("genres")),
        "themes": names(a.get("themes")),
        "synopsis": (a.get("synopsis") or "").replace("\n", " ").strip(),
        "type": a.get("type") or "",
        "episodes": a.get("episodes") or "",
        "score": a.get("score") or "",
        "year": year or "",
        "studio": names(a.get("studios")),
        "members": a.get("members") or "",
    }


def fetch_page(session: requests.Session, page: int, retries: int = 5) -> dict:
    for attempt in range(retries):
        resp = session.get(API, params={"page": page, "limit": 25}, timeout=30)
        if resp.status_code == 429:  # rate limited: back off and retry
            time.sleep(2 + attempt * 2)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Page {page} kept returning 429; try again later.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", type=int, default=40, help="number of pages (25 titles each)")
    parser.add_argument("--out", type=Path, default=Path("data/anime.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with requests.Session() as session, args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for page in range(1, args.pages + 1):
            payload = fetch_page(session, page)
            for anime in payload.get("data", []):
                row = to_row(anime)
                if row["title"] and row["title"] not in seen:
                    seen.add(row["title"])
                    writer.writerow(row)
            print(f"page {page}/{args.pages}: {len(seen)} titles")
            if not payload.get("pagination", {}).get("has_next_page"):
                break
            time.sleep(1.1)
    print(f"Wrote {len(seen)} titles to {args.out}")


if __name__ == "__main__":
    main()
