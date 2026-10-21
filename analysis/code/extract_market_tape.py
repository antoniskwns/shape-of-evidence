"""
Build data/processed/polymarket_match_odds.csv from the Polymarket order-book files.

Reads data/raw/markets.parquet and snapshots.parquet (Marketlens dataset, see the
thesis bibliography) and writes the one-minute price series used by
decision_pipeline.py. Needs pandas and pyarrow; the pipeline does not.

Four markets are used:

    Will Spain win on 2026-07-19?              resolved No
    Will Spain vs. Argentina end in a draw?    resolved Yes
    Will Argentina win on 2026-07-19?          resolved No
    Spain vs. Argentina: Team to Advance       resolved Spain

The first three settle on the score after ninety minutes (0-0), the last on
the eventual winner (Spain, after extra time); see Section 4.1.

Near settlement one side of the book empties and the midpoint is missing, so
the remaining quote is used instead. Each series is cut at its own resolution
time, since stale quotes are recorded afterwards. Times are clock time, so
the goal in the 106th minute of play appears around minute 162.

Run:  python extract_market_tape.py
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
QUESTIONS = {
    "spain":     "Will Spain win on 2026-07-19?",
    "draw":      "Will Spain vs. Argentina end in a draw?",
    "argentina": "Will Argentina win on 2026-07-19?",
    "advance":   "Spain vs. Argentina: Team to Advance",
}
END_MIN = 200.0


def main():
    markets = pd.read_parquet(RAW / "markets.parquet")
    snaps = pd.read_parquet(RAW / "snapshots.parquet")
    kickoff = int(markets[markets.question == QUESTIONS["draw"]].iloc[0].open_time_ms)

    def series(question):
        row = markets[markets.question == question].iloc[0]
        sub = snaps[(snaps.market_id == row.market_id)
                    & (snaps.collected_at_ms <= row.resolved_at_ms)]
        sub = sub.sort_values("collected_at_ms")
        t = np.asarray((sub.collected_at_ms - kickoff) / 60000.0, dtype=float)
        mid = np.asarray(sub.midpoint, dtype=float)
        bid = np.asarray(sub.best_bid, dtype=float)
        ask = np.asarray(sub.best_ask, dtype=float)
        price = np.where(np.isnan(mid),
                         np.where(np.isnan(ask), bid,
                                  np.where(np.isnan(bid), ask, mid)), mid)
        keep = ~np.isnan(price)
        print(f"  {question[:44]:46s} resolved {row.winning_outcome:>9s}"
              f"  final {price[keep][-1]:.4f}")
        return t[keep], price[keep]

    print("markets used:")
    raw = {k: series(q) for k, q in QUESTIONS.items()}
    grid = np.arange(0.0, END_MIN + 0.5, 1.0)
    interp = {k: np.interp(grid, t, p) for k, (t, p) in raw.items()}

    result = np.vstack([interp["spain"], interp["draw"], interp["argentina"]])
    result = result / result.sum(0)          # normalise the three-way market
    adv_spain = np.clip(interp["advance"], 0.0, 1.0)

    base = pd.Timestamp("2026-07-19 19:00:00+00:00")
    out = PROCESSED / "polymarket_match_odds.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "spain", "draw", "argentina",
                    "elapsed_min", "spain_advance", "argentina_advance"])
        for i, g in enumerate(grid):
            w.writerow([(base + pd.Timedelta(minutes=float(g))).isoformat(),
                        f"{result[0, i]:.8f}", f"{result[1, i]:.8f}",
                        f"{result[2, i]:.8f}", f"{g:.8f}",
                        f"{adv_spain[i]:.8f}", f"{1 - adv_spain[i]:.8f}"])
    print(f"\nwrote {len(grid)} rows -> {out}")


if __name__ == "__main__":
    main()
