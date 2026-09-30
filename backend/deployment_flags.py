"""Source-code switches. Environment variables override these defaults.

Leave both update switches False while reviewing/testing the application.
None makes PUBLIC_PRECOMPUTED_ONLY automatic: True on Railway/production,
False for local development. Set it explicitly to select another default.
"""

PUBLIC_PRECOMPUTED_ONLY: bool | None = None
MONTHLY_UPDATES_ENABLED = False
MODAL_MONTHLY_SCHEDULE_ENABLED = False
MONTHLY_UPDATE_CRON = "0 3 1 * *"  # First day of each calendar month, 03:00 UTC.
MONTHLY_UPDATE_MAX_NEW_PAPERS = 300
