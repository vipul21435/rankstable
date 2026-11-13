"""Download the Open LLM Leaderboard results and write them as a wide CSV for rankstable.

The source is the dataset ``open-llm-leaderboard/contents`` on Hugging Face
(https://huggingface.co/datasets/open-llm-leaderboard/contents). Its page states no licence,
so this repository contains this script and numbers derived from the table, not the table.

    python examples/fetch_open_llm_leaderboard.py open_llm_leaderboard.csv

One row per evaluated model (the dataset's ``eval_name``), one column per benchmark, holding
the normalised scores that the leaderboard itself averages. Only the standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DATASET = "open-llm-leaderboard/contents"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
BENCHMARKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
PAGE = 100


def fetch_page(offset: int, attempts: int = 5) -> dict:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": PAGE,
        }
    )
    request = urllib.request.Request(
        f"{ROWS_URL}?{query}", headers={"User-Agent": "rankstable-example"}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == attempts - 1:
                raise
            wait = 2**attempt
            print(f"offset {offset}: {error}; retrying in {wait} s", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="path of the CSV to write")
    args = parser.parse_args()

    rows: list[dict] = []
    total = None
    while total is None or len(rows) < total:
        page = fetch_page(len(rows))
        total = page["num_rows_total"]
        batch = [item["row"] for item in page["rows"]]
        if not batch:
            break
        rows.extend(batch)
        print(f"fetched {len(rows)} of {total} rows", file=sys.stderr)

    seen: set[str] = set()
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *BENCHMARKS])
        for row in rows:
            name = row["eval_name"]
            if name in seen:
                continue
            seen.add(name)
            writer.writerow([name, *("" if row[b] is None else row[b] for b in BENCHMARKS)])
    print(f"wrote {len(seen)} models and {len(BENCHMARKS)} benchmarks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
