"""
Hypothetical small B2B SaaS basket tracker.

Builds an equal-weight (EW) and cap-weight (CW) basket of small B2B SaaS stocks,
hypothetical 1x and 2x daily-reset inverse series on each, and IGV as a benchmark.
Writes CSVs to ./data. Everything is hypothetical and backtested before FORWARD_START.
Not a fund, not investment advice, not tradable.

Raw stock prices are NOT written to disk (Yahoo data terms); only derived index values.
"""
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
START = "2024-01-01"
BENCH = "IGV"
FORWARD_START = "2026-11-30"   # close of this day = 100 for the forward record
ANNUAL_FEE = 0.0095            # reference: QID 95 bps
BORROW = 0.00                  # borrow-cost assumption, scaled by leverage
DAYS = 252
OUT = "data"

BASKET = {
    # horizontal
    "WK": "horizontal", "FRSH": "horizontal", "RAMP": "horizontal",
    "AMPL": "horizontal", "ASAN": "horizontal", "FIVN": "horizontal",
    "PRGS": "horizontal", "CXM": "horizontal", "BL": "horizontal",
    "PD": "horizontal", "NABL": "horizontal", "MITK": "horizontal",
    "YEXT": "horizontal", "SPT": "horizontal", "ASUR": "horizontal",
    # vertical
    "QTWO": "vertical", "INTA": "vertical", "AGYS": "vertical",
    "VERX": "vertical", "SPSC": "vertical", "NCNO": "vertical",
    "ALKT": "vertical", "EVCM": "vertical", "PAR": "vertical",
    "WEAV": "vertical", "LAW": "vertical",
    # cybersecurity
    "TENB": "cybersecurity", "RPD": "cybersecurity", "OSPN": "cybersecurity",
}
# Acquired and delisted. Kept in the history while they traded; they drop out
# of the baskets automatically once their prices stop. Not replaced.
ACQUIRED = {"RAMP", "WEAV"}


# ---------------------------------------------------------------- data in
def load_prices():
    import yfinance as yf

    import time

    tickers = list(BASKET) + [BENCH]
    # threads=False avoids parallel downloads colliding on yfinance's local cache
    raw = yf.download(tickers, start=START, auto_adjust=True, progress=False,
                      threads=False)["Close"]

    # retry any ticker that came back empty, one at a time
    for t in tickers:
        if t in ACQUIRED:
            continue
        for attempt in range(3):
            if t in raw.columns and not raw[t].dropna().empty:
                break
            time.sleep(2 * (attempt + 1))
            try:
                one = yf.download(t, start=START, auto_adjust=True, progress=False,
                                  threads=False)["Close"]
                if isinstance(one, pd.DataFrame):
                    one = one.iloc[:, 0]
                if not one.dropna().empty:
                    raw[t] = one
            except Exception as exc:
                print(f"retry {attempt + 1} failed for {t}: {exc}")
    return raw


def drop_partial_today(df):
    """Drop today's row if the run happens before the data has settled after the close."""
    now = pd.Timestamp.now(tz="America/New_York")
    if df.index[-1].date() == now.date() and now.hour < 17:
        df = df.iloc[:-1]
    return df


def load_shares(prices):
    """Share counts for cap weights. Falls back to the last saved value."""
    import yfinance as yf

    path = os.path.join(OUT, "shares.csv")
    old = pd.read_csv(path, index_col=0)["shares"] if os.path.exists(path) else pd.Series(dtype=float)
    new = {}
    for t in prices.columns:
        val = np.nan
        try:
            info = yf.Ticker(t).info
            val = info.get("sharesOutstanding") or np.nan
            if np.isnan(val) and info.get("marketCap"):
                px = prices[t].dropna()
                if len(px):
                    val = info["marketCap"] / px.iloc[-1]
        except Exception as exc:  # network or Yahoo hiccup: keep going
            print(f"warning: share count failed for {t}: {exc}")
        new[t] = val
    shares = pd.Series(new, dtype=float).fillna(old.reindex(prices.columns))
    shares.rename("shares").to_frame().to_csv(path)
    return shares


# ---------------------------------------------------------------- build
def build(prices, bench, shares):
    """Return (vals, returns). Mirrors the notebook logic."""
    returns = prices.pct_change(fill_method=None)
    bench_ret = bench.pct_change(fill_method=None)

    ew_ret = returns.mean(axis=1).dropna()

    caps = prices.shift(1).mul(shares, axis=1)
    w = caps.where(returns.notna())
    w = w.div(w.sum(axis=1), axis=0)
    cw_ret = (w * returns).sum(axis=1, min_count=1).dropna()

    def inverse(ret, leverage):
        return -leverage * ret - (ANNUAL_FEE + leverage * BORROW) / DAYS

    def value(ret):
        v = 100 * (1 + ret).cumprod()
        return pd.concat([pd.Series([100.0], index=[prices.index[0]]), v])

    vals = pd.DataFrame({
        "EW basket": value(ew_ret),
        "CW basket": value(cw_ret),
        "EW 1x inverse": value(inverse(ew_ret, 1)),
        "EW 2x inverse": value(inverse(ew_ret, 2)),
        "CW 1x inverse": value(inverse(cw_ret, 1)),
        "CW 2x inverse": value(inverse(cw_ret, 2)),
        "IGV": value(bench_ret.dropna()),
    }).dropna()
    return vals, returns


def forward_record(vals):
    """Values rebased to 100 at the last close on or before FORWARD_START."""
    fs = pd.Timestamp(FORWARD_START)
    before = vals.loc[:fs]
    if before.empty or vals.index[-1] <= before.index[-1]:
        return None  # forward period has not started yet
    base_date = before.index[-1]
    fwd = vals.loc[base_date:]
    return fwd / fwd.iloc[0] * 100


def window_table(vals):
    last = vals.index[-1]
    starts = {
        "1M": last - pd.DateOffset(months=1),
        "3M": last - pd.DateOffset(months=3),
        "6M": last - pd.DateOffset(months=6),
        "Since Jan 1, 2026": pd.Timestamp("2025-12-31"),
        "Full period": vals.index[0],
    }
    fs = pd.Timestamp(FORWARD_START)
    if vals.index[-1] > fs:
        starts["Since forward start"] = fs
    rows = {}
    for name, s in starts.items():
        base = vals.loc[:s]
        base = base.iloc[-1] if len(base) else vals.iloc[0]
        rows[name] = (vals.iloc[-1] / base - 1) * 100
    return pd.DataFrame(rows).T.round(1)


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)

    raw = drop_partial_today(load_prices())

    # health checks: fail loudly instead of publishing bad numbers
    last_date = raw.index[-1]
    age = (pd.Timestamp.now().normalize() - last_date.normalize()).days
    if age > 5:
        sys.exit(f"Stale data: last row is {last_date.date()} ({age} days old). Not publishing.")
    missing = [t for t in list(BASKET) + [BENCH]
               if t not in raw.columns or raw[t].dropna().empty]
    bad = [t for t in missing if t not in ACQUIRED]
    if bad:
        sys.exit(f"No data for: {bad}. Not publishing.")
    if missing:
        print(f"note: no price history returned for acquired names {missing}; "
              "history will exclude them and differ slightly from a full backtest.")

    prices = raw[[t for t in BASKET if t in raw.columns]].dropna(axis=1, how="all")
    bench = raw[BENCH]
    shares = load_shares(prices)

    vals, returns = build(prices, bench, shares)

    vals.rename_axis("date").round(4).to_csv(os.path.join(OUT, "saas_tracker_values.csv"))
    fwd = forward_record(vals)
    if fwd is not None:
        fwd.rename_axis("date").round(4).to_csv(os.path.join(OUT, "saas_tracker_forward.csv"))
    summary = window_table(vals)
    summary.rename_axis("window").to_csv(os.path.join(OUT, "saas_tracker_summary.csv"))

    live = int(returns.iloc[-1].notna().sum())
    status = pd.DataFrame([{
        "last_price_date": vals.index[-1].date(),
        "live_names": live,
        "run_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M"),
    }])
    status.to_csv(os.path.join(OUT, "saas_tracker_status.csv"), index=False)

    print(f"Last price date: {vals.index[-1].date()}  live names: {live}")
    print("Total return by window (%):")
    print(summary.T)


if __name__ == "__main__":
    main()
