"""
Optional long-running daily scheduler for nse_scanner.py.

Use this ONLY if you can't set up an OS-level cron job (Linux/macOS) or
Task Scheduler entry (Windows) — which is the more robust way to run this
daily (see the SCHEDULING notes at the bottom of nse_scanner.py).

This script simply stays alive and calls the scanner every day at a fixed
time. It must itself be running continuously (e.g. on a machine or server
that's always on) for this to work.

Run with:  python daily_runner.py
"""

import time
import schedule

from nse_scanner import main as run_scan

# Time to run each day (24-hour format, local machine time).
# 18:00 = after NSE market close (15:30 IST) with a comfortable buffer
# for the day's data to be available from Yahoo Finance.
RUN_TIME = "18:00"

# Only run on weekdays (NSE is closed Sat/Sun)
WEEKDAYS_ONLY = True


def job():
    print(f"Triggering scheduled scan at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        run_scan()
    except Exception as e:
        print(f"Scan failed: {e}")


if WEEKDAYS_ONLY:
    schedule.every().monday.at(RUN_TIME).do(job)
    schedule.every().tuesday.at(RUN_TIME).do(job)
    schedule.every().wednesday.at(RUN_TIME).do(job)
    schedule.every().thursday.at(RUN_TIME).do(job)
    schedule.every().friday.at(RUN_TIME).do(job)
else:
    schedule.every().day.at(RUN_TIME).do(job)

print(f"Daily runner started. Will scan every weekday at {RUN_TIME} (local time).")
print("Leave this process running. Press Ctrl+C to stop.")

while True:
    schedule.run_pending()
    time.sleep(30)
