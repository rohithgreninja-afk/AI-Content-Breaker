"""Offline checks for the optional immutable Parquet range transport."""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m0_parquet import HTTPRangeReader, ParquetWindows, conversion_urls


class FakeResponse:
    def __init__(self, payload, start, end, total, status=206):
        self.payload = payload
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}
        self.consumed = False

    def iter_content(self, chunk_size):
        self.consumed = True
        yield self.payload

    def close(self):
        pass


class FakeSession:
    def __init__(self, files):
        self.files = files
        self.requests = []
        self.status = 206

    def get(self, url, *, headers, **kwargs):
        data = self.files[url]
        requested = headers["Range"]
        self.requests.append((url, requested))
        if requested.startswith("bytes=-"):
            start, end = len(data) - int(requested[7:]), len(data) - 1
        else:
            start, end = map(int, requested[6:].split("-"))
        self.response = FakeResponse(data[start:end + 1], start, end,
                                     len(data), self.status)
        return self.response


class TestHTTPRanges(unittest.TestCase):
    def test_random_access_and_cache_use_exact_ranges(self):
        session = FakeSession({"file": b"abcdefghiPAR1"})
        reader = HTTPRangeReader("file", session=session)
        reader.seek(1)
        self.assertEqual(reader.read(3), b"bcd")
        self.assertEqual(session.requests[-1], ("file", "bytes=1-3"))
        count = len(session.requests)
        reader.seek(2)
        self.assertEqual(reader.read(1), b"c")
        self.assertEqual(len(session.requests), count)
        reader.seek(-4, io.SEEK_END)
        self.assertEqual(reader.read(), b"PAR1")

    def test_server_ignoring_range_is_rejected_without_downloading(self):
        session = FakeSession({"file": b"abcdefghiPAR1"})
        session.status = 200
        with self.assertRaisesRegex(ValueError, "did not honor"):
            HTTPRangeReader("file", session=session)
        self.assertFalse(session.response.consumed)

    def test_read_budget_prevents_accidental_whole_shard_download(self):
        session = FakeSession({"file": b"a" * 1000 + b"PAR1"})
        reader = HTTPRangeReader("file", session=session, max_request_bytes=100)
        with self.assertRaisesRegex(ValueError, "budget"):
            reader.read()
        self.assertEqual(len(session.requests), 1)


class TestParquetWindows(unittest.TestCase):
    def test_windows_preserve_global_order_across_row_groups_and_shards(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        files = {}
        for shard in range(2):
            sink = pa.BufferOutputStream()
            table = pa.table({"id": list(range(shard * 5, shard * 5 + 5)),
                              "answers": [[str(i)] for i in range(shard * 5, shard * 5 + 5)],
                              "unneeded": ["ignored"] * 5})
            pq.write_table(table, sink, row_group_size=2)
            files[f"shard{shard}"] = sink.getvalue().to_pybytes()
        session = FakeSession(files)
        windows = ParquetWindows(list(files), ["id", "answers"],
                                  session=session, expected_rows=10)
        result = windows.get_rows(3, 5)
        self.assertEqual(result["num_rows_total"], 10)
        self.assertEqual([row["row_idx"] for row in result["rows"]], list(range(3, 8)))
        self.assertEqual([row["row"]["id"] for row in result["rows"]], list(range(3, 8)))
        self.assertEqual(result["rows"][0]["row"]["answers"], ["3"])
        self.assertNotIn("unneeded", result["rows"][0]["row"])
        complete = ParquetWindows(list(files), columns=None, session=session)
        self.assertEqual(complete.get_rows(0, 1)["rows"][0]["row"],
                         {"id": 0, "answers": ["0"], "unneeded": "ignored"})
        with self.assertRaises(ValueError):
            windows.get_rows(9, 2)

    def test_conversion_requires_unchanged_source_revision(self):
        session = Mock()
        session.get.return_value.json.return_value = {"sha": "changed"}
        with self.assertRaisesRegex(ValueError, "source revision changed"):
            conversion_urls("Hello-SimpleAI/HC3", "all", "train", "expected",
                            session=session)
        self.assertEqual(session.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
