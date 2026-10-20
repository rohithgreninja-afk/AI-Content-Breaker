"""Dataset correctness checks: revision, resume, balance and split lineage."""

import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import m0_dataset as dataset


class MapperTests(unittest.TestCase):
    def test_mage_labels_are_explicit(self):
        row = {"src": "eli5_human", "text": "prose", "label": 1}
        self.assertEqual(dataset.map_mage(row)[0][0], "human")
        row.update(src="eli5_machine_continuation_7B", label=0)
        self.assertEqual(dataset.map_mage(row)[0][:2], ("ai", "llama_7B"))
        with self.assertRaises(ValueError):
            dataset.map_mage({**row, "label": 2})

    def test_hc3_lists_and_raid_unknowns(self):
        self.assertEqual(dataset._first('["answer", "other"]'), "answer")
        self.assertEqual(dataset._first([None, "answer"]), "answer")
        self.assertIsNone(dataset._first(None))
        self.assertEqual(dataset.map_raid({"attack": "none", "model": None,
                                          "domain": "news", "generation": "words"}), [])

    def test_null_text_is_not_stringified(self):
        rows = [{"row_idx": 0, "id": 5, "wiki_intro": None,
                 "generated_intro": "machine " * 30}]
        self.assertEqual(dataset.collect("wiki", rows), [])

    def test_natural_source_ids_and_global_prefixes(self):
        self.assertEqual(dataset.natural_group("raid", {"source_id": "original"}, "a"),
                         dataset.natural_group("raid", {"source_id": "original"}, "b"))
        opening = " ".join(f"word{i}" for i in range(20))
        groups, links = {}, []
        dataset.collect("wiki", [{"row_idx": 0, "id": 8,
                                 "wiki_intro": opening + " human ending",
                                 "generated_intro": opening + " machine ending"}], groups, links)
        self.assertEqual(len(links), 2)
        self.assertEqual(links[0][1], links[1][1])


class PairTests(unittest.TestCase):
    @staticmethod
    def pair(number, human, machine):
        return [[f"h{number}", "human", "human", "wiki:wikipedia", human],
                [f"a{number}", "model", "ai", "wiki:wikipedia", machine]]

    def test_deduplication_drops_whole_pair_and_keeps_lengths(self):
        rows = self.pair(1, "human one", "machine one")
        rows += self.pair(2, " HUMAN   ONE ", "machine two")
        groups = {row[0]: row[0] for row in rows}
        kept, row_groups, dropped = dataset.deduplicate_pairs(rows, groups)
        self.assertEqual((len(kept), dropped), (2, 1))
        self.assertEqual(row_groups["h1"], row_groups["a1"])
        self.assertEqual({row[2] for row in kept}, {"human", "ai"})

    def test_connected_natural_groups_and_unselected_prefix_chain(self):
        rows = self.pair(1, "human one", "machine one")
        rows += self.pair(2, "human two", "machine two")
        groups = {row[0]: row[0] for row in rows}
        links = [("h1", "prefix:x"), ("unselected", "prefix:x"),
                 ("unselected", "prefix:y"), ("a2", "prefix:y")]
        _, row_groups, _ = dataset.deduplicate_pairs(rows, groups, links)
        self.assertEqual(len(set(row_groups.values())), 1)

    def test_invalid_pairs_fail(self):
        rows = self.pair(1, "human", "machine two")
        with self.assertRaises(ValueError):
            dataset.deduplicate_pairs(rows, {row[0]: row[0] for row in rows})

    def test_csv_contract_and_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "sample.csv")
            rows = self.pair(1, "human one", "machine one")
            dataset.write_csv(rows, path)
            with open(path, encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                self.assertEqual(next(reader), ["id", "model", "label", "domain", "text"])
                self.assertEqual(list(reader), rows)
            self.assertEqual(len(dataset.file_sha256(path)), 64)


class FetchTests(unittest.TestCase):
    def test_only_incomplete_trailing_record_can_be_recovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pool.jsonl"
            path.write_bytes(b'{"row_idx": 1}\n{"row_idx":')
            self.assertEqual(dataset._read_pool(str(path)), [{"row_idx": 1}])
            self.assertEqual(path.read_bytes(), b'{"row_idx": 1}\n')
            path.write_bytes(b'{"row_idx":}\n{"row_idx": 2}\n')
            with self.assertRaisesRegex(RuntimeError, "corrupt cached JSONL"):
                dataset._read_pool(str(path))

    def test_rows_api_rejects_unpinned_revision(self):
        response = Mock(status_code=200, headers={"x-revision": "changed"})
        with patch.object(dataset.requests, "get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "does not match pinned"):
                dataset._get_rows("repo", "default", "train", 0, 1, "pinned")

    def test_resume_requires_complete_block_not_just_first_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = ("repo", "default", "train", "pinned", 1, dataset.map_mage)
            offset = random.Random(dataset.SEED).sample(range(0, 500 - dataset.BLOCK_LEN), 1)[0]
            pool = Path(temporary) / "mage_pool.jsonl"
            pool.write_text(json.dumps({"row_idx": offset}) + "\n", encoding="utf-8")
            identity = {"dataset": "repo", "config": "default", "split": "train",
                        "revision": "pinned", "num_rows_total": 500, "partial": False,
                        "seed": dataset.SEED, "block_length": dataset.BLOCK_LEN,
                        "completed_offsets": []}
            (Path(temporary) / "mage_pool.meta.json").write_text(json.dumps(identity), encoding="utf-8")
            calls = []

            def get_rows(*args):
                _, _, _, start, length, revision = args
                calls.append((start, length, revision))
                return {"num_rows_total": 500, "rows": [
                    {"row_idx": index, "row": {"text": "words"}}
                    for index in range(start, start + length)]}

            with patch.object(dataset, "DATA_DIR", temporary), \
                    patch.dict(dataset.SOURCES, {"mage": source}), \
                    patch.object(dataset, "_get_rows", side_effect=get_rows), \
                    patch.object(dataset.time, "sleep"):
                result = dataset.fetch_pool("mage")
            self.assertEqual(len(result), dataset.BLOCK_LEN)
            self.assertIn((offset, dataset.BLOCK_LEN, "pinned"), calls)
            metadata = json.loads((Path(temporary) / "mage_pool.meta.json").read_text())
            self.assertEqual(metadata["completed_offsets"], [offset])
            self.assertEqual(metadata["pool_sha256"], dataset.file_sha256(str(pool)))

    def test_legacy_incomplete_pool_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "mage_pool.jsonl").write_text('{"row_idx": 0}\n')
            with patch.object(dataset, "DATA_DIR", temporary), \
                    patch.object(dataset, "_get_rows", return_value={"num_rows_total": 319071}):
                with self.assertRaisesRegex(RuntimeError, "complete seeded sample"):
                    dataset.fetch_pool("mage")


if __name__ == "__main__":
    unittest.main()
