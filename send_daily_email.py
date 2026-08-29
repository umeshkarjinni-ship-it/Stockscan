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
import io
import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage

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

# Inline price charts for the top BUY signals.
#
# Each chart shows Close, the Volatility Stop and KAMA on the signal's own
# timeframe, with BUY/SELL markers — so you can see the setup rather than
# only reading numbers. Capped because every chart means one more download
# and a larger message.
INCLUDE_CHARTS = os.environ.get("SCANNER_EMAIL_CHARTS", "1") not in ("0", "false", "False")
MAX_CHARTS = int(os.environ.get("SCANNER_MAX_CHARTS", "6"))
CHART_BARS = 60          # bars of history to display

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
                except (TypeError, ValueError):
                    pass
                # Flag on BAR AGE, not drift size — see build_dashboard.py.
                # A Monthly bar closing 30 days ago is not a defect, and
                # tagging it STALE trained the reader to ignore the flag.
                try:
                    bar_date = pd.to_datetime(r.get("Date"))
                    age = int((pd.Timestamp.now().normalize() - bar_date.normalize()).days)
                    if age > 10:
                        val += ('<span style="background:#fdeaea;color:#b3261e;font:600 9px sans-serif;'
                                f'padding:1px 4px;border-radius:2px">BAR {age}d OLD</span>')
                except Exception:
                    pass
            # BUY rows only. On a SELL row the PULLBACK badge contradicted
            # the row it sat on.
            if c == "PctFrom52WHigh" and in_pullback and kind == "buy":
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



# ---------------------------------------------------------------------
# Price charts
# ---------------------------------------------------------------------

def build_charts(buy_df):
    """
    Render a small PNG per BUY signal: Close, Volatility Stop, KAMA, with
    BUY/SELL markers on the signal's own timeframe.

    Returns {cid: png_bytes}. Failures are skipped rather than raised — a
    chart that won't render must never stop the email going out.
    """
    if buy_df is None or not INCLUDE_CHARTS:
        return {}

    # Imported here so the email still sends on a machine without
    # matplotlib, or if a backend problem arises.
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless: no display needed
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception as e:
        print(f"matplotlib unavailable — skipping charts ({e})")
        return {}

    try:
        from nse_scanner import fetch_single, compute_signals, resample, fetch_nifty_daily
    except Exception as e:
        print(f"Could not import scanner functions — skipping charts ({e})")
        return {}

    nifty_close = None
    try:
        nd = fetch_nifty_daily()
        if nd is not None and not nd.empty:
            nifty_close = nd["Close"]
    except Exception:
        pass

    charts = {}
    for _, r in buy_df.head(MAX_CHARTS).iterrows():
        symbol = str(r.get("Symbol", "")).strip()
        tf = str(r.get("Timeframe", "Weekly")).strip()
        if not symbol:
            continue
        try:
            raw = fetch_single(symbol)
            if raw is None or len(raw) == 0:
                continue
            rule = "ME" if tf.lower() == "monthly" else "W"
            sig = compute_signals(resample(raw.sort_index(), rule), nifty_close)
            if sig is None or sig.empty:
                continue

            view = sig.tail(CHART_BARS)

            fig, ax = plt.subplots(figsize=(4.6, 2.4), dpi=110)
            ax.plot(view.index, view["Close"], lw=1.6, color="#1f77b4", label="Close")
            if "VSTOP" in view.columns:
                ax.plot(view.index, view["VSTOP"], lw=1.0, color="#ff7f0e",
                        ls="--", label="VStop")
            if "KAMA_MID" in view.columns:
                ax.plot(view.index, view["KAMA_MID"], lw=1.0, color="#7f7f7f",
                        alpha=0.8, label="KAMA")

            # Signal markers
            if "BUY_SIGNAL_CORE" in view.columns:
                b = view[view["BUY_SIGNAL_CORE"].fillna(False).astype(bool)]
                if not b.empty:
                    ax.scatter(b.index, b["Close"], marker="^", s=55,
                               color="#2ca02c", zorder=5, label="BUY")
            if "SELL_SIGNAL" in view.columns:
                sl = view[view["SELL_SIGNAL"].fillna(False).astype(bool)]
                if not sl.empty:
                    ax.scatter(sl.index, sl["Close"], marker="v", s=55,
                               color="#d62728", zorder=5, label="SELL")

            # Shade the pullback zone, the one validated marker.
            try:
                lookback = 52 if rule == "W" else 12
                hi = sig["Close"].rolling(lookback, min_periods=10).max()
                off = (sig["Close"] - hi) / hi * 100
                pb = (sig["Close"] > sig["KAMA_MID"]) & (sig["RSI"] < 45) & (off <= -8)
                pbv = pb.reindex(view.index).fillna(False).astype(bool)
                if pbv.any():
                    ax.fill_between(view.index, view["Close"].min(), view["Close"].max(),
                                    where=pbv.values, color="#2ca02c", alpha=0.10,
                                    step="mid", label="Pullback")
            except Exception:
                pass

            ax.set_title(f"{symbol} — {tf}", fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.25, lw=0.5)
            ax.legend(fontsize=5.5, loc="upper left", framealpha=0.85)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            fig.autofmt_xdate(rotation=0, ha="center")
            fig.tight_layout(pad=0.4)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            charts[f"chart_{symbol}"] = buf.getvalue()
        except Exception as e:
            print(f"  chart failed for {symbol}: {e}")
            continue

    if charts:
        total_kb = sum(len(v) for v in charts.values()) / 1024
        print(f"Rendered {len(charts)} chart(s), {total_kb:.0f} KB total")
    return charts


def build_html(charts=None):
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

    # Charts, referenced by Content-ID so they render inline rather than
    # as separate downloads.
    if charts:
        parts.append('<h3 style="font:600 15px sans-serif;color:#1a1a1a;margin:24px 0 8px">'
                     'Charts</h3>')
        parts.append('<table style="border-collapse:collapse"><tr>')
        for i, cid in enumerate(charts):
            if i and i % 2 == 0:
                parts.append('</tr><tr>')
            parts.append(f'<td style="padding:4px"><img src="cid:{cid}" '
                         f'style="width:330px;max-width:100%;border:1px solid #eee;'
                         f'border-radius:4px" alt="{esc(cid)}"></td>')
        parts.append('</tr></table>')
        parts.append('<div style="font:11px sans-serif;color:#777;margin-top:6px;line-height:1.6">'
                     'Blue = close, orange dashed = volatility stop, grey = KAMA trend. '
                     'Green shading marks the pullback zone.</div>')

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

    buy_for_charts = read_first_available(BUY_FILE, BUY_FALLBACK)
    charts = build_charts(buy_for_charts)

    html = build_html(charts)
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

    # "related" wraps the HTML together with the images it references by
    # CID; that pairing is what makes them display inline.
    related = MIMEMultipart("related")
    body = MIMEMultipart("alternative")
    body.attach(MIMEText(html, "html"))
    related.attach(body)

    for cid, png in (charts or {}).items():
        img = MIMEImage(png, _subtype="png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        related.attach(img)

    msg.attach(related)

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
