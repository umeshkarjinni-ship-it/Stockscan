"""
send_daily_email.py
=====================
Emails the day's BUY/SELL recommendations — the same data the dashboard
shows — as a self-contained HTML message.

WHY A SEPARATE SCRIPT
---------------------
nse_scanner.py already has an email path, but it fires at the end of the
scan and reports the scanner's own view. This one runs AFTER the whole
pipeline (scan -> report -> rank -> conflicts -> tracker), so it can
include things the scanner doesn't know yet: the ranked buy list, price
staleness, conflicting signals, market regime, and open paper positions.

It reads the CSVs, so it never re-downloads anything and takes seconds.

CONFIGURATION
-------------
Uses the same environment variables as the scanner:

    SCANNER_EMAIL_FROM          sending Gmail address
    SCANNER_EMAIL_TO            recipient (comma-separate for several)
    SCANNER_EMAIL_APP_PASSWORD  Gmail App Password, NOT your login password

If any are missing it exits quietly, so it's safe to leave in a workflow
before you've set the secrets.

USAGE
-----
    python send_daily_email.py
"""

import os
import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

import pandas as pd

SIGNALS_DIR = "signals"
BUY_FILE = os.path.join(SIGNALS_DIR, "daily_top20_buy_ranked.csv")
BUY_FALLBACK = os.path.join(SIGNALS_DIR, "daily_top20_buy.csv")
SELL_FILE = os.path.join(SIGNALS_DIR, "daily_top20_sell.csv")
CONFLICTS_FILE = os.path.join(SIGNALS_DIR, "conflicting_signals.csv")
REGIME_FILE = os.path.join(SIGNALS_DIR, "market_regime.json")
PAPER_FILE = os.path.join(SIGNALS_DIR, "paper_trades.csv")

EMAIL_FROM = os.environ.get("SCANNER_EMAIL_FROM", "")
EMAIL_TO = os.environ.get("SCANNER_EMAIL_TO", "")
EMAIL_APP_PASSWORD = os.environ.get("SCANNER_EMAIL_APP_PASSWORD", "")

SMTP_SERVER = os.environ.get("SCANNER_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SCANNER_SMTP_PORT", "587"))

# Optional: link to your published dashboard, shown at the bottom.
DASHBOARD_URL = os.environ.get("SCANNER_DASHBOARD_URL", "")

# Attach the generated dashboard so it can be opened and rendered directly.
#
# GitHub Pages needs a PUBLIC repo on the free plan, so a private repo has no
# published dashboard URL — and linking to the file on github.com just shows
# the HTML source, not the rendered page. Attaching sidesteps both problems.
DASHBOARD_FILE = os.path.join("docs", "index.html")
ATTACH_DASHBOARD = os.environ.get("SCANNER_ATTACH_DASHBOARD", "1") not in ("0", "false", "False")

MAX_ROWS = 15


def read_first_available(*paths):
    """
    First readable, non-empty CSV from the given paths.

    NOTE: `read_csv_safe(a) or read_csv_safe(b)` does NOT work here —
    pandas raises "truth value of a DataFrame is ambiguous" on `or`.
    """
    for p in paths:
        df = read_csv_safe(p)
        if df is not None:
            return df
    return None


def read_csv_safe(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if not df.empty:
                return df
        except Exception:
            pass
    return None


def esc(v):
    return "" if v is None else str(v).replace("<", "&lt;").replace(">", "&gt;")


def fmt(v, dec=2):
    try:
        f = float(v)
        if pd.isna(f):
            return "—"
        return f"{f:,.{dec}f}"
    except (TypeError, ValueError):
        return "—" if v is None or (isinstance(v, float) and pd.isna(v)) else esc(v)


def signal_table(df, title, kind):
    if df is None:
        return (f'<h3 style="font:600 15px sans-serif;color:#1a1a1a;margin:24px 0 8px">{esc(title)}</h3>'
                f'<p style="font:13px sans-serif;color:#777;margin:0">None today.</p>')

    accent = "#0a7d3a" if kind == "buy" else "#b3261e"
    cols = [c for c in ["Symbol", "Timeframe", "Price", "CurrentPrice", "PriceDriftPct",
                        "PctFrom52WHigh", "BuyScore", "RSI", "ADX"] if c in df.columns]
    labels = {"Price": "Signal", "CurrentPrice": "Current", "PriceDriftPct": "Drift %",
              "PctFrom52WHigh": "Off 52w High", "BuyScore": "Score"}

    head = "".join(
        f'<th style="text-align:left;padding:6px 10px;border-bottom:2px solid #ddd;'
        f'font:600 11px sans-serif;color:#666;text-transform:uppercase">'
        f'{esc(labels.get(c, c))}</th>'
        for c in cols
    )

    body = ""
    for _, r in df.head(MAX_ROWS).iterrows():
        in_pullback = str(r.get("PullbackZone", "")).strip().lower() in ("true", "1")
        cells = ""
        for c in cols:
            val = fmt(r[c]) if c not in ("Symbol", "Timeframe") else esc(r[c])
            style = "padding:6px 10px;border-bottom:1px solid #eee;font:13px monospace;color:#222"
            if c == "Symbol":
                style += f";font-weight:700;color:{accent}"
            if c == "PriceDriftPct":
                try:
                    d = float(r[c])
                    style += f";color:{'#0a7d3a' if d >= 0 else '#b3261e'}"
                    if abs(d) >= 7:
                        val += ' <span style="background:#fdeaea;color:#b3261e;font:600 9px sans-serif;' \
                               'padding:1px 4px;border-radius:2px">STALE</span>'
                except (TypeError, ValueError):
                    pass
            if c == "PctFrom52WHigh" and in_pullback:
                val += ' <span style="background:#e6f6ec;color:#0a7d3a;font:600 9px sans-serif;' \
                       'padding:1px 4px;border-radius:2px">PULLBACK</span>'
            cells += f'<td style="{style}">{val}</td>'
        body += f"<tr>{cells}</tr>"

    return (
        f'<h3 style="font:600 15px sans-serif;color:#1a1a1a;margin:24px 0 8px">'
        f'{esc(title)} <span style="font-weight:400;color:#888">({len(df)})</span></h3>'
        f'<table style="border-collapse:collapse;width:100%">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        + (
            '<div style="font:11px sans-serif;color:#777;margin-top:8px;line-height:1.6">'
            '<b style="color:#0a7d3a">PULLBACK</b> = RSI under 45, 8%+ below the 52-week high, '
            'trend intact. Out-of-sample testing found pullback entries beat strength entries '
            'by ~7-12%, while the strength entry this scanner fires on tested <i>worse</i> than '
            'a random nearby date. Use it to time an entry you have already decided on — '
            'not as a reason to buy.</div>'
            if kind == "buy" and "PctFrom52WHigh" in df.columns else ""
        )
    )


def build_html():
    buy = read_first_available(BUY_FILE, BUY_FALLBACK)
    sell = read_csv_safe(SELL_FILE)
    conflicts = read_csv_safe(CONFLICTS_FILE)
    paper = read_csv_safe(PAPER_FILE)

    regime = None
    if os.path.exists(REGIME_FILE):
        try:
            with open(REGIME_FILE, encoding="utf-8") as f:
                regime = json.load(f)
        except Exception:
            regime = None

    today = datetime.now().strftime("%d %b %Y")
    parts = [
        '<div style="max-width:720px;margin:0 auto;padding:20px;'
        'font-family:-apple-system,Segoe UI,sans-serif;background:#fff">',
        f'<div style="font:700 20px sans-serif;color:#1a1a1a">NiftyPulsePro — Daily Signals</div>',
        f'<div style="font:13px sans-serif;color:#888;margin-top:4px">{today} '
        '&nbsp;·&nbsp; Research tool, not financial advice</div>',
    ]

    if regime:
        wk, mo = regime.get("Weekly", "?"), regime.get("Monthly", "?")
        note = ""
        if regime.get("weekly_buys_blocked"):
            note = ('<div style="font:12px sans-serif;color:#8a6d00;margin-top:6px">'
                    'Weekly BUY signals are suppressed by the market-regime filter — '
                    'intentional, not missing data.</div>')
        parts.append(
            '<div style="background:#f7f7f9;border:1px solid #e3e3e8;border-radius:6px;'
            'padding:12px 14px;margin-top:16px">'
            '<span style="font:600 12px sans-serif;color:#555">NIFTY REGIME</span><br>'
            f'<span style="font:13px monospace;color:#333">Weekly '
            f'<b style="color:{"#0a7d3a" if wk == "UP" else "#b3261e"}">{esc(wk)}</b>'
            f'&nbsp;&nbsp;Monthly '
            f'<b style="color:{"#0a7d3a" if mo == "UP" else "#b3261e"}">{esc(mo)}</b></span>'
            f"{note}</div>"
        )

    if conflicts is not None and not conflicts.empty:
        rows = "".join(
            f'<li style="font:13px sans-serif;color:#333;margin:2px 0">'
            f'<b>{esc(r.get("Symbol"))}</b>: BUY {esc(r.get("BuyTimeframe"))} @ {fmt(r.get("BuyPrice"))} '
            f'vs SELL {esc(r.get("SellTimeframe"))} @ {fmt(r.get("SellPrice"))}</li>'
            for _, r in conflicts.iterrows()
        )
        parts.append(
            '<div style="background:#fdeaea;border:1px solid #f5c2c0;border-radius:6px;'
            'padding:12px 14px;margin-top:12px">'
            '<span style="font:600 12px sans-serif;color:#b3261e">CONFLICTING SIGNALS</span>'
            f'<ul style="margin:6px 0 0;padding-left:18px">{rows}</ul></div>'
        )

    parts.append(signal_table(buy, "BUY Signals", "buy"))
    parts.append(signal_table(sell, "SELL Signals", "sell"))

    if paper is not None and "Status" in paper.columns:
        open_n = int((paper["Status"] == "OPEN").sum())
        closed = paper[paper["Status"] == "CLOSED"]
        line = f"{open_n} open position(s)"
        if len(closed):
            ret = pd.to_numeric(closed["ReturnPct"], errors="coerce").dropna()
            if len(ret):
                line += (f" · {len(ret)} closed, win rate "
                         f"{(ret > 0).mean() * 100:.0f}%, avg {ret.mean():+.2f}%")
        parts.append(
            '<div style="margin-top:24px;padding-top:14px;border-top:1px solid #eee">'
            '<span style="font:600 12px sans-serif;color:#555">PAPER TRACKER</span><br>'
            f'<span style="font:13px sans-serif;color:#333">{esc(line)}</span></div>'
        )

    if DASHBOARD_URL:
        parts.append(
            f'<div style="margin-top:16px"><a href="{esc(DASHBOARD_URL)}" '
            'style="font:13px sans-serif;color:#1a5fb4">Open dashboard on GitHub →</a></div>'
        )

    parts.append(
        '<div style="margin-top:24px;padding-top:14px;border-top:1px solid #eee;'
        'font:11px sans-serif;color:#999;line-height:1.6">'
        'Signals are pure price/volume technicals with no fundamental or macro context. '
        'Prices marked STALE have moved significantly since the signal bar was completed. '
        'Validate independently before acting on anything here.</div></div>'
    )

    return "".join(parts)


def main():
    if not (EMAIL_FROM and EMAIL_TO and EMAIL_APP_PASSWORD):
        print("Email not configured — set SCANNER_EMAIL_FROM / _TO / _APP_PASSWORD. Skipping.")
        return

    html = build_html()
    recipients = [a.strip() for a in EMAIL_TO.split(",") if a.strip()]

    buy = read_first_available(BUY_FILE, BUY_FALLBACK)
    sell = read_csv_safe(SELL_FILE)
    subject = (f"NiftyPulsePro — {len(buy) if buy is not None else 0} BUY / "
               f"{len(sell) if sell is not None else 0} SELL — "
               f"{datetime.now().strftime('%d %b %Y')}")

    # "mixed" rather than "alternative" — an alternative container is for
    # different renderings of the SAME content, and attachments in one are
    # unreliable across mail clients.
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(recipients)

    body = MIMEMultipart("alternative")
    body.attach(MIMEText(html, "html"))
    msg.attach(body)

    if ATTACH_DASHBOARD and os.path.exists(DASHBOARD_FILE):
        try:
            with open(DASHBOARD_FILE, "rb") as f:
                part = MIMEApplication(f.read(), _subtype="html")
            stamp = datetime.now().strftime("%Y-%m-%d")
            part.add_header("Content-Disposition", "attachment",
                            filename=f"niftypulsepro_dashboard_{stamp}.html")
            msg.attach(part)
            size_kb = os.path.getsize(DASHBOARD_FILE) / 1024
            print(f"Attached dashboard ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"Could not attach dashboard: {e}")
    elif ATTACH_DASHBOARD:
        print(f"{DASHBOARD_FILE} not found — sending without the attachment.")

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        server.sendmail(EMAIL_FROM, recipients, msg.as_string())

    print(f"Email sent to {len(recipients)} recipient(s): {subject}")


if __name__ == "__main__":
    main()
