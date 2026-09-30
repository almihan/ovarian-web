"""Offline checks for the bundled Cell Ontology resources and provenance."""

import json
import unittest

from backend.cellexlink_lite.resources import (
    CELL_ONTOLOGY_RELEASE,
    CELL_ONTOLOGY_RESOURCE_VERSION,
    DEFAULT_ONTOLOGY_PATH,
)
from backend.services.cell_hierarchy import CellHierarchyIndex


class BundledOntologyTests(unittest.TestCase):
    def test_june_lexicon_contains_canonical_cell_terms(self):
        with DEFAULT_ONTOLOGY_PATH.open(encoding="utf-8") as source:
            terms = {
                row["norm_concept_id"]: row
                for row in (json.loads(line) for line in source if line.strip())
            }
        self.assertEqual(CELL_ONTOLOGY_RELEASE, "2026-06-08")
        self.assertEqual(DEFAULT_ONTOLOGY_PATH.name, "cell_ontology_v2026-06-08.jsonl")
        self.assertEqual(CELL_ONTOLOGY_RESOURCE_VERSION, "cell-ontology-v2026-06-08")
        self.assertEqual(terms["CL:0000000"]["norm_preferred_label"], "cell")
        self.assertEqual(
            terms["CL:0000624"]["norm_preferred_label"],
            "CD4-positive, alpha-beta T cell",
        )
        self.assertGreater(len(terms), 3300)

    def test_hierarchy_keeps_its_actual_release_and_working_paths(self):
        index = CellHierarchyIndex()
        source = index.source()
        self.assertEqual(source["release"], "2025-12-17")
        self.assertEqual(source["lexicon_release"], "2026-06-08")
        self.assertFalse(source["hierarchy_matches_lexicon"])
        paths = index.get_hierarchy_paths("CL:0000624")
        self.assertTrue(paths)
        for path in paths:
            self.assertEqual(path[0], "CL:0000000")
            self.assertEqual(path[-1], "CL:0000624")


if __name__ == "__main__":
    unittest.main()
