"""Preview or trigger the private update on an already running controller.

Preview (ID discovery only): python scripts/trigger_monthly_update.py
Commit (may use GPU/API):    python scripts/trigger_monthly_update.py --commit
Status:                     python scripts/trigger_monthly_update.py --status
Requires PUBLIC_BASE_URL and MONTHLY_UPDATE_TOKEN; updates must be enabled.
"""

import argparse
import json
import os

import requests
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--commit", action="store_true", help="Process and publish new papers; default is a discovery-only preview.")
    action.add_argument("--status", action="store_true", help="Read the private updater status.")
    parser.add_argument("--url", default=os.getenv("PUBLIC_BASE_URL", ""), help="Controller base URL (defaults to PUBLIC_BASE_URL).")
    args = parser.parse_args()
    token = os.getenv("MONTHLY_UPDATE_TOKEN", "").strip()
    if not args.url or not token:
        parser.error("Set PUBLIC_BASE_URL (or --url) and MONTHLY_UPDATE_TOKEN.")
    url = args.url.rstrip("/") + "/api/internal/corpus-updates"
    headers = {"X-Corpus-Update-Token": token}
    if args.status:
        response = requests.get(url, headers=headers, timeout=30)
    else:
        response = requests.post(url, headers=headers, json={"dry_run": not args.commit}, timeout=30)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
