"""Offline checks for durable corpus growth and cached PMID selection."""

import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.services import corpus_store as store


def _append_in_process(root, seeds, corpus_id, start, ready):
    store.settings = SimpleNamespace(precomputed_corpora_dir=Path(root), data_dir=Path(root).parent)
    store.SEED_CORPUS_ROOT = Path(seeds)
    ready.wait(5)
    store.add_prediction_rows(corpus_id, [{"pmid": str(value), "chunks": []} for value in range(start, start + 20)])


class CorpusStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.seeds = self.base / "seeds"
        self.seeds.mkdir()
        self.persistent = self.base / "persistent"
        self.ids = list(store.CORPUS_FILES)
        self._write_seed(self.ids[0], [{"canonical_id": "pmid:123", "chunks": []}])
        self._write_seed(self.ids[1], [{"doc_key": "456", "chunks": []}])
        self.addCleanup(patch.stopall)
        patch.object(store, "SEED_CORPUS_ROOT", self.seeds).start()
        patch.object(store, "settings", SimpleNamespace(
            precomputed_corpora_dir=self.persistent,
            data_dir=self.base / "data",
        )).start()

    def _write_seed(self, corpus_id, rows):
        path = self.seeds / store.CORPUS_FILES[corpus_id]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _stored_rows(self, corpus_id):
        path = self.persistent / store.CORPUS_FILES[corpus_id]
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_union_index_and_restart_keep_updates_without_overwriting_seeds(self):
        self.assertEqual(store.get_known_pmids(), {"123", "456"})
        seed_bytes = (self.seeds / store.CORPUS_FILES[self.ids[0]]).read_bytes()
        result = store.add_prediction_rows(self.ids[0], [
            {"pmid": "789", "chunks": []},
            {"pmid": "789", "chunks": [{"text": "duplicate"}]},
            {"pmid": "123", "chunks": [{"text": "reviewed result must stay"}]},
        ])
        self.assertEqual((result["added"], result["skipped"]), (1, 2))
        self.assertEqual(result["known_pmid_count"], 3)
        # A fresh startup must reuse the store instead of copying seed data over it.
        store.ensure_corpus_store()
        self.assertEqual(store.get_known_pmids(), {"123", "456", "789"})
        self.assertEqual(len(self._stored_rows(self.ids[0])), 2)
        self.assertEqual((self.seeds / store.CORPUS_FILES[self.ids[0]]).read_bytes(), seed_bytes)
        index = json.loads((self.persistent / store.INDEX_FILENAME).read_text())
        self.assertEqual(index["pmids"], ["123", "456", "789"])

    def test_copy_existing_paper_to_second_topic_needs_no_new_prediction(self):
        store.get_known_pmids()
        self.assertEqual(store.get_corpus_pmids(self.ids[0]), {"123"})
        self.assertEqual(store.get_corpus_pmids(self.ids[1]), {"456"})
        destination = self.base / "membership.jsonl"
        store.select_pmid_predictions(["123"], destination)
        result = store.add_prediction_rows(self.ids[1], store._rows(destination))
        self.assertEqual(result["added"], 1)
        self.assertEqual(store.get_known_pmids(), {"123", "456"})
        self.assertEqual(store.get_corpus_pmids(self.ids[1]), {"123", "456"})
        self.assertEqual(store.add_prediction_rows(self.ids[1], store._rows(destination))["added"], 0)

    def test_selects_both_topics_preserves_rows_and_reports_missing(self):
        first = {"pmid": "123", "chunk_id": 1, "chunks": [{"text": "first"}]}
        second = {"pmid": "123", "chunk_id": 2, "chunks": [{"text": "second"}]}
        self._write_seed(self.ids[0], [first, first, second])
        self._write_seed(self.ids[1], [
            {"pmid": "123", "chunks": [{"text": "other corpus"}]},
            {"pmid": "456", "chunks": []},
        ])
        destination = self.base / "subset.jsonl"
        result = store.select_pmid_predictions(["456", "PMID:123", "999", "123"], destination)
        self.assertEqual(result["requested_pmids"], ["456", "123", "999"])
        self.assertEqual(result["found_pmids"], ["456", "123"])
        self.assertEqual(result["missing_pmids"], ["999"])
        self.assertEqual(result["paper_count"], 2)
        rows = list(store._rows(destination))
        self.assertEqual([store.prediction_pmid(row) for row in rows], ["456", "123", "123"])
        self.assertEqual(rows[1:], [first, second])

    def test_index_recovers_after_corpus_commit_before_index_commit(self):
        self.assertEqual(store.get_known_pmids(), {"123", "456"})
        path = self.persistent / store.CORPUS_FILES[self.ids[0]]
        # Model a process that committed a corpus, then crashed before writing its
        # derived PMID index. The changed source signature forces index repair.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"pmid": "790", "chunks": []}) + "\n")
        self.assertEqual(store.get_known_pmids(), {"123", "456", "790"})
        (self.persistent / store.INDEX_FILENAME).write_text("broken json", encoding="utf-8")
        self.assertEqual(store.get_known_pmids(), {"123", "456", "790"})

    def test_failed_atomic_commit_keeps_previous_corpus(self):
        store.ensure_corpus_store()
        path = self.persistent / store.CORPUS_FILES[self.ids[0]]
        original = path.read_bytes()
        replace = store.os.replace

        def fail_corpus_commit(source, destination):
            if Path(destination) == path:
                raise OSError("simulated interrupted commit")
            return replace(source, destination)

        with patch.object(store.os, "replace", side_effect=fail_corpus_commit):
            with self.assertRaises(OSError):
                store.add_prediction_rows(self.ids[0], [{"pmid": "790", "chunks": []}])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(store.get_known_pmids(), {"123", "456"})
        self.assertEqual(list(self.persistent.glob("*.tmp")), [])

    def test_invalid_new_rows_do_not_partially_commit(self):
        store.ensure_corpus_store()
        for invalid in ({"pmid": "PMC123"}, {"title": "missing id"}, {"pmid": "790", "value": float("nan")}):
            with self.assertRaises(ValueError):
                store.add_prediction_rows(self.ids[0], [{"pmid": "789", "chunks": []}, invalid])
        self.assertEqual(store.get_known_pmids(), {"123", "456"})

    def test_pmid_identity_fields_and_normalization(self):
        for row in ({"pmid": 123}, {"canonical_id": "pmid:123"}, {"doc_key": "123"}, {"id": "123"}):
            self.assertEqual(store.prediction_pmid(row), "123")
        self.assertEqual(store.normalize_pmid("https://pubmed.ncbi.nlm.nih.gov/123/"), "123")
        self.assertEqual(store.normalize_pmid("000123"), "123")
        for value in ("PMC123", "0", True, "12.3", "foo", None):
            self.assertEqual(store.normalize_pmid(value), "")

    def test_selection_cannot_replace_managed_source(self):
        store.ensure_corpus_store()
        destination = self.persistent / store.CORPUS_FILES[self.ids[0]]
        original = destination.read_bytes()
        with self.assertRaises(ValueError):
            store.select_pmid_predictions(["123"], destination)
        self.assertEqual(destination.read_bytes(), original)

    def test_parallel_processes_do_not_lose_committed_papers(self):
        store.ensure_corpus_store()
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        workers = [context.Process(target=_append_in_process, args=(
            str(self.persistent), str(self.seeds), self.ids[0], start, ready,
        )) for start in (1000, 2000)]
        try:
            for worker in workers:
                worker.start()
            ready.set()
            for worker in workers:
                worker.join(10)
                self.assertEqual(worker.exitcode, 0)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(2)
        expected = {"123", "456"} | {str(value) for start in (1000, 2000) for value in range(start, start + 20)}
        self.assertEqual(store.get_known_pmids(), expected)
        self.assertEqual(len(self._stored_rows(self.ids[0])), 41)


if __name__ == "__main__":
    unittest.main()
