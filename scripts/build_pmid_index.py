"""Build/refresh the union PMID list without retrieval, GPU, or API calls.

Usage:
    python scripts/build_pmid_index.py
    python scripts/build_pmid_index.py --store-dir data/precomputed_corpora
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, help="Override PRECOMPUTED_CORPORA_DIR for this command.")
    args = parser.parse_args()
    from backend.services import corpus_store

    if args.store_dir is not None:
        corpus_store.settings = replace(
            corpus_store.settings,
            precomputed_corpora_dir=args.store_dir.expanduser().resolve(),
        )
    root = corpus_store.ensure_corpus_store()
    known = corpus_store.get_known_pmids()
    print(json.dumps({"index_file": str(root / corpus_store.INDEX_FILENAME), "known_pmid_count": len(known)}, indent=2))


if __name__ == "__main__":
    main()
