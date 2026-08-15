"""
build_dashboard.py
====================
Turns the day's CSV/TXT outputs into a single self-contained HTML page —
so you can check results by opening one file (or a GitHub Pages URL)
instead of digging through signals/*.csv and *.txt.

Reads (all optional — sections are skipped gracefully if a file is
missing):
    signals/daily_top20_buy_ranked.csv
    signals/daily_top20_sell.csv
    signals/paper_trades.csv
    signals/paper_trade_summary.txt
    signals/backtest_report.txt

Writes:
    docs/index.html

USAGE
-----
    python build_dashboard.py

Run it any time after the other scripts (rank_buy.py, paper_trade_tracker.py,
etc.) to refresh the dashboard. The GitHub Actions workflow does this
automatically after every daily scan — see .github/workflows/daily-scan.yml.

VIEWING IT
----------
- Locally: just double-click docs/index.html to open it in a browser.
- Online: enable GitHub Pages (Settings -> Pages -> Source: "Deploy from
  a branch" -> Branch: main, folder: /docs) and it'll be published at
  https://<username>.github.io/<repo>/ automatically.
"""

import os
from datetime import datetime

import pandas as pd

SIGNALS_DIR = "signals"
DOCS_DIR = "docs"
OUTPUT_FILE = os.path.join(DOCS_DIR, "index.html")

BUY_RANKED_FILE = os.path.join(SIGNALS_DIR, "daily_top20_buy_ranked.csv")
BUY_ML_RANKED_FILE = os.path.join(SIGNALS_DIR, "daily_top20_buy_ml_ranked.csv")
CONFLICTS_FILE = os.path.join(SIGNALS_DIR, "conflicting_signals.csv")
SELL_FILE = os.path.join(SIGNALS_DIR, "daily_top20_sell.csv")
PAPER_TRADES_FILE = os.path.join(SIGNALS_DIR, "paper_trades.csv")
PAPER_SUMMARY_FILE = os.path.join(SIGNALS_DIR, "paper_trade_summary.txt")
BACKTEST_REPORT_FILE = os.path.join(SIGNALS_DIR, "backtest_report.txt")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_csv_safe(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if not df.empty:
                return df
        except Exception:
            pass
    return None


def read_text_safe(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return None


def fmt_num(value, decimals=2, suffix=""):
    try:
        v = float(value)
        if pd.isna(v):
            return "—"
        return f"{v:.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def pct_class(value):
    try:
        v = float(value)
        if v > 0:
            return "pos"
        if v < 0:
            return "neg"
    except (TypeError, ValueError):
        pass
    return ""


def esc(value):
    return "" if value is None else str(value).replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------

def build_ticker(buy_df, sell_df):
    items = []
    if buy_df is not None:
        for _, r in buy_df.head(15).iterrows():
            items.append(f'<span class="tick buy">▲ {esc(r.get("Symbol",""))} <b>{fmt_num(r.get("Price"))}</b></span>')
    if sell_df is not None:
        for _, r in sell_df.head(15).iterrows():
            items.append(f'<span class="tick sell">▼ {esc(r.get("Symbol",""))} <b>{fmt_num(r.get("Price"))}</b></span>')
    if not items:
        items = ['<span class="tick">No signals yet — run the scanner to populate this dashboard.</span>']
    strip = "".join(items)
    return f'<div class="ticker-wrap"><div class="ticker">{strip}{strip}</div></div>'


def build_stat_cards(buy_df, sell_df, paper_df):
    buy_count = len(buy_df) if buy_df is not None else 0
    sell_count = len(sell_df) if sell_df is not None else 0

    open_count = closed_count = win_rate = avg_return = None
    if paper_df is not None:
        open_count = int((paper_df["Status"] == "OPEN").sum())
        closed = paper_df[paper_df["Status"] == "CLOSED"]
        closed_count = len(closed)
        if closed_count > 0:
            returns = pd.to_numeric(closed["ReturnPct"], errors="coerce").dropna()
            if len(returns) > 0:
                win_rate = (returns > 0).mean() * 100
                avg_return = returns.mean()

    cards = [
        ("Today's BUY signals", str(buy_count), ""),
        ("Today's SELL signals", str(sell_count), ""),
        ("Open paper positions", str(open_count) if open_count is not None else "—", ""),
        ("Closed paper trades", str(closed_count) if closed_count is not None else "—", ""),
        ("Paper win rate", fmt_num(win_rate, 1, "%") if win_rate is not None else "—", pct_class(win_rate) if win_rate is not None else ""),
        ("Paper avg return", fmt_num(avg_return, 2, "%") if avg_return is not None else "—", pct_class(avg_return) if avg_return is not None else ""),
    ]

    html = ""
    for label, value, cls in cards:
        html += f'<div class="card"><div class="card-label">{esc(label)}</div><div class="card-value {cls}">{esc(value)}</div></div>'
    return html


def build_signal_table(df, title, kind):
    if df is None:
        return f'<section class="panel"><h2>{esc(title)}</h2><p class="empty">No data yet — this file hasn\'t been generated this run.</p></section>'

    cols = [c for c in ["Symbol", "Name", "Category", "Timeframe", "Price", "CurrentPrice", "PriceDriftPct", "BuyScore", "MLWinProbability", "RSI", "ADX", "VolumeRatio"] if c in df.columns]
    rows_html = ""
    for _, r in df.head(20).iterrows():
        cells = ""
        for c in cols:
            val = fmt_num(r[c]) if c in ('Price','CurrentPrice','PriceDriftPct','BuyScore','MLWinProbability','RSI','ADX','VolumeRatio') else r[c]
            # Highlight signals whose price has already moved a long way
            # from the signal bar — the entry you'd get now differs from
            # the one the signal identified.
            if c == "PriceDriftPct":
                cls = pct_class(r[c])
                stale = bool(r.get("StalePrice", False))
                badge = ' <span class="badge flag">STALE</span>' if stale else ""
                cells += f"<td class='{cls}'>{esc(val)}{badge}</td>"
            else:
                cells += f"<td>{esc(val)}</td>"
        rows_html += f"<tr>{cells}</tr>"

    # Friendlier headers — "Price" alone is ambiguous now that both the
    # signal-bar price and the current price are shown.
    header_labels = {
        "Price": "Signal Price",
        "CurrentPrice": "Current Price",
        "PriceDriftPct": "Drift %",
        "VolumeRatio": "Vol Ratio",
        "MLWinProbability": "ML Win %",
    }
    header_html = "".join(f"<th>{esc(header_labels.get(c, c))}</th>" for c in cols)
    row_class = "buy-row" if kind == "buy" else "sell-row"
    return f'''
    <section class="panel">
      <h2>{esc(title)}</h2>
      <div class="table-wrap">
        <table class="{row_class}">
          <thead><tr>{header_html}</tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </section>'''


def build_paper_table(df):
    if df is None:
        return '<section class="panel"><h2>Paper Trade Tracker</h2><p class="empty">No tracked positions yet — the tracker opens its first positions after the next daily scan.</p></section>'

    df = df.copy()
    df["_sort"] = (df["Status"] == "OPEN").astype(int)
    df = df.sort_values(["_sort", "EntryDate"], ascending=[False, False])

    rows_html = ""
    for _, r in df.iterrows():
        status = r.get("Status", "")
        if status == "OPEN":
            ret = r.get("UnrealizedReturnPct", "")
            ret_label = f'{fmt_num(ret, 2, "%")} <span class="badge open">OPEN</span>'
        else:
            ret = r.get("ReturnPct", "")
            ret_label = f'{fmt_num(ret, 2, "%")} <span class="badge closed">CLOSED</span>'
        cls = pct_class(ret)
        flag = str(r.get("Flag", "") or "").strip()
        flag_html = f' <span class="badge flag" title="{esc(flag)}">⚠ CHECK DATA</span>' if flag else ""
        rows_html += (
            f"<tr><td>{esc(r.get('Symbol',''))}</td><td>{esc(r.get('Timeframe',''))}</td>"
            f"<td>{esc(r.get('EntryDate',''))}</td><td>{fmt_num(r.get('EntryPrice',''))}</td>"
            f"<td>{esc(r.get('HoldingDays',''))}</td>"
            f"<td class='{cls}'>{ret_label}{flag_html}</td></tr>"
        )

    return f'''
    <section class="panel">
      <h2>Paper Trade Tracker <span class="subhead">— today's picks, tracked forward</span></h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Symbol</th><th>Timeframe</th><th>Entry Date</th><th>Entry Price</th><th>Days Held</th><th>Return</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </section>'''


def build_conflicts_panel(df):
    if df is None or df.empty:
        return ""

    rows_html = ""
    for _, r in df.iterrows():
        rows_html += (
            f"<tr><td>{esc(r.get('Symbol',''))}</td>"
            f"<td class='pos'>{esc(r.get('BuyTimeframe',''))} @ {fmt_num(r.get('BuyPrice',''))}</td>"
            f"<td class='neg'>{esc(r.get('SellTimeframe',''))} @ {fmt_num(r.get('SellPrice',''))}</td></tr>"
        )

    return f'''
    <section class="panel conflict-panel">
      <h2>⚠ Conflicting Signals <span class="subhead">— BUY on one timeframe, SELL on another, same day</span></h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Symbol</th><th>BUY signal</th><th>SELL signal</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </section>'''


def build_backtest_panel(text):
    if not text:
        return '<section class="panel"><h2>Backtest Summary</h2><p class="empty">No backtest report yet — run the "Run Backtest" workflow to generate one.</p></section>'
    lines = [l for l in text.splitlines() if ":" in l and "=" not in l]
    rows = ""
    for line in lines:
        k, _, v = line.partition(":")
        rows += f'<div class="stat"><span>{esc(k.strip())}</span><b>{esc(v.strip())}</b></div>'
    return f'<section class="panel"><h2>Backtest Summary</h2><div class="stat-grid">{rows}</div></section>'


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    buy_df = read_csv_safe(BUY_ML_RANKED_FILE)
    if buy_df is None:
        buy_df = read_csv_safe(BUY_RANKED_FILE)
    sell_df = read_csv_safe(SELL_FILE)
    paper_df = read_csv_safe(PAPER_TRADES_FILE)
    backtest_text = read_text_safe(BACKTEST_REPORT_FILE)
    conflicts_df = read_csv_safe(CONFLICTS_FILE)

    generated = datetime.now().strftime("%d %b %Y, %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NiftyPulsePro Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,500;0,600;1,500&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #10141B;
    --panel: #171D27;
    --border: #262E3A;
    --text: #E8EAED;
    --muted: #8B93A1;
    --gold: #D4A24C;
    --pos: #4ADE80;
    --neg: #F87171;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    padding-bottom: 60px;
  }}
  h1, h2 {{ font-family: 'Newsreader', serif; font-weight: 600; margin: 0; }}
  .pos {{ color: var(--pos); }}
  .neg {{ color: var(--neg); }}
  .mono {{ font-family: 'IBM Plex Mono', monospace; }}

  .ticker-wrap {{
    background: #0B0E13;
    border-bottom: 1px solid var(--border);
    overflow: hidden;
    white-space: nowrap;
    padding: 10px 0;
  }}
  .ticker {{
    display: inline-block;
    animation: scroll 40s linear infinite;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
  }}
  .ticker:hover {{ animation-play-state: paused; }}
  .tick {{ margin: 0 28px; color: var(--muted); }}
  .tick.buy {{ color: var(--pos); }}
  .tick.sell {{ color: var(--neg); }}
  .tick b {{ color: var(--text); }}
  @keyframes scroll {{
    0% {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
  }}
  @media (prefers-reduced-motion: reduce) {{ .ticker {{ animation: none; }} }}

  header {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 24px 24px;
  }}
  header h1 {{ font-size: 32px; letter-spacing: 0.3px; }}
  header .eyebrow {{
    color: var(--gold);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  header .meta {{ color: var(--muted); font-size: 14px; margin-top: 8px; }}

  .cards {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 24px 32px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
  }}
  .card-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .card-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 24px; font-weight: 600; }}

  main {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 24px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }}
  .panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px 24px;
  }}
  .panel h2 {{ font-size: 20px; margin-bottom: 14px; }}
  .panel h2 .subhead {{ font-family: 'IBM Plex Sans', sans-serif; font-weight: 400; font-size: 13px; color: var(--muted); }}
  .empty {{ color: var(--muted); font-size: 14px; margin: 0; }}

  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{
    text-align: left;
    color: var(--muted);
    font-weight: 500;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  td {{
    padding: 9px 12px;
    border-bottom: 1px solid #1C222C;
    font-family: 'IBM Plex Mono', monospace;
    white-space: nowrap;
  }}
  tbody tr:hover {{ background: #1B222D; }}
  .buy-row tbody tr td:first-child {{ color: var(--pos); font-weight: 600; }}
  .sell-row tbody tr td:first-child {{ color: var(--neg); font-weight: 600; }}

  .badge {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 3px;
    margin-left: 6px;
    letter-spacing: 0.5px;
  }}
  .badge.open {{ background: rgba(212,162,76,0.15); color: var(--gold); }}
  .badge.closed {{ background: rgba(139,147,161,0.15); color: var(--muted); }}
  .badge.flag {{ background: rgba(248,113,113,0.15); color: var(--neg); cursor: help; }}
  .conflict-panel {{ border-color: rgba(248,113,113,0.35); }}

  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
  .stat {{
    display: flex;
    justify-content: space-between;
    background: #10151D;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 13px;
  }}
  .stat span {{ color: var(--muted); }}
  .stat b {{ font-family: 'IBM Plex Mono', monospace; }}

  footer {{
    max-width: 1100px;
    margin: 32px auto 0;
    padding: 0 24px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
  }}
</style>
</head>
<body>

{build_ticker(buy_df, sell_df)}

<header>
  <div class="eyebrow">NSE / BSE · Weekly &amp; Monthly Trend Scanner</div>
  <h1>NiftyPulsePro Dashboard</h1>
  <div class="meta">Last updated {esc(generated)} &nbsp;·&nbsp; Research tool, not financial advice</div>
</header>

<div class="cards">
{build_stat_cards(buy_df, sell_df, paper_df)}
</div>

<main>
{build_conflicts_panel(conflicts_df)}
{build_signal_table(buy_df, "Today's Top BUY Signals", "buy")}
{build_signal_table(sell_df, "Today's Top SELL Signals", "sell")}
{build_paper_table(paper_df)}
{build_backtest_panel(backtest_text)}
</main>

<footer>
  Generated automatically by build_dashboard.py from the latest signals/ output.
  Signals are pure price/volume technicals with no fundamental or macro context —
  validate before acting on anything shown here.
</footer>

</body>
</html>"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
