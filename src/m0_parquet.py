"""Read deterministic row windows from immutable Hugging Face Parquet shards.

Recovery transport for a datasets-server outage. It requests byte ranges
for Parquet metadata and selected row groups, never a complete remote file.
The conversion commit is separate from the source commit. Both revisions are
pinned and representative rows are checked against rows API samples. A matching
current source SHA alone does not establish a cryptographic association.
"""

from __future__ import annotations

import bisect
import io
import re
import time
from collections import OrderedDict
from urllib.parse import quote

import requests


PINNED_CONVERSIONS = {
    "Hello-SimpleAI/HC3": "7444b1cb1469246e7b5d191b03f69b7e7c951134",
    "aadityaubhat/GPT-wiki-intro": "5f831a4d91cfd3cd4eb7cb5acdd7a880be43c78e",
    "liamdugan/raid": "5008b2d167f74bfd4ceff6b3b0512b7097d7cbb1",
}
SPLIT_ROWS = {"Hello-SimpleAI/HC3": 24322,
              "aadityaubhat/GPT-wiki-intro": 150000, "liamdugan/raid": 2270000}
_READERS = {}


class HTTPRangeReader(io.RawIOBase):
    """Seekable, bounded HTTP reader that rejects servers ignoring Range."""

    def __init__(self, url: str, session=None, timeout: float = 45,
                 max_request_bytes: int = 32 * 1024 * 1024):
        super().__init__()
        self.url = url
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_request_bytes = max_request_bytes
        self.position = 0
        self.bytes_downloaded = 0
        self.requests_made = 0
        self._cached: OrderedDict[tuple[int, int], bytes] = OrderedDict()
        self.size = None
        trailer, start, total = self._request("bytes=-8")
        self.size = total
        if len(trailer) != 8 or trailer[-4:] != b"PAR1":
            raise ValueError("remote file is not an ordinary Parquet file")
        self._cached[(start, total)] = trailer

    def _request(self, byte_range: str):
        for attempt in range(4):
            response = None
            try:
                response = self.session.get(
                    self.url, headers={"Range": byte_range, "Accept-Encoding": "identity"},
                    timeout=self.timeout, stream=True)
                self.requests_made += 1
                if response.status_code != 206:
                    if response.status_code in (429, 500, 502, 503, 504):
                        raise requests.HTTPError(f"range server returned {response.status_code}")
                    raise ValueError(f"server did not honor byte range: {response.status_code}")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                     response.headers.get("Content-Range", ""))
                if not match:
                    raise ValueError("range response lacks valid Content-Range")
                start, inclusive_end, total = map(int, match.groups())
                expected = inclusive_end - start + 1
                if expected <= 0 or expected > self.max_request_bytes:
                    raise ValueError("remote byte range exceeds the request budget")
                if self.size is not None and total != self.size:
                    raise ValueError("remote file size changed during ranged reading")
                if byte_range.startswith("bytes=-"):
                    if inclusive_end + 1 != total or expected != int(byte_range[7:]):
                        raise ValueError("server returned the wrong suffix range")
                elif byte_range != f"bytes={start}-{inclusive_end}":
                    raise ValueError("server returned a different byte range")
                chunks, received = [], 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    received += len(chunk)
                    self.bytes_downloaded += len(chunk)
                    if received > expected:
                        raise ValueError("range response body exceeds Content-Range")
                    chunks.append(chunk)
                if received != expected:
                    raise requests.ConnectionError("truncated byte range response")
                return b"".join(chunks), start, total
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
            finally:
                if response is not None:
                    response.close()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = (offset if whence == io.SEEK_SET else
                    self.position + offset if whence == io.SEEK_CUR else
                    self.size + offset if whence == io.SEEK_END else None)
        if position is None or position < 0:
            raise ValueError("invalid seek")
        self.position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        end = self.size if size < 0 else min(self.position + size, self.size)
        start = self.position
        if end <= start:
            return b""
        if end - start > self.max_request_bytes:
            raise ValueError("read exceeds the HTTP range request budget")
        for (cached_start, cached_end), data in list(self._cached.items()):
            if cached_start <= start and end <= cached_end:
                self.position = end
                return data[start - cached_start:end - cached_start]
        data, _, _ = self._request(f"bytes={start}-{end - 1}")
        self._cached[(start, end)] = data
        # Bound cached compressed data, independently of the remote shard size.
        while sum(map(len, self._cached.values())) > 8 * 1024 * 1024:
            self._cached.popitem(last=False)
        self.position = end
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


class ParquetWindows:
    """Read global row windows across shards listed in their original order."""

    def __init__(self, urls: list[str], columns: list[str] | None = None, *, session=None,
                 expected_rows: int | None = None):
        import pyarrow.parquet as pq

        self.columns = columns
        self.files, self.readers, self.starts = [], [], [0]
        self._row_groups = OrderedDict()
        def open_shard(url):
            reader = HTTPRangeReader(url, session=session)
            parquet = pq.ParquetFile(reader, pre_buffer=False, buffer_size=0)
            missing = set(columns or []) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"Parquet shard lacks columns: {sorted(missing)}")
            return reader, parquet

        if session is None and len(urls) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as pool:
                opened = list(pool.map(open_shard, urls))
        else:
            opened = [open_shard(url) for url in urls]
        for reader, parquet in opened:
            self.readers.append(reader)
            self.files.append(parquet)
            self.starts.append(self.starts[-1] + parquet.metadata.num_rows)
        self.num_rows = self.starts[-1]
        if expected_rows is not None and self.num_rows != expected_rows:
            raise ValueError(f"Parquet has {self.num_rows} rows; expected {expected_rows}")

    def get_rows(self, offset: int, length: int) -> dict:
        """Return the rows-API shape with exact global indices and nested values."""
        if offset < 0 or length < 0 or offset + length > self.num_rows:
            raise ValueError("row window lies outside the converted split")
        rows = []
        cursor, end = offset, offset + length
        while cursor < end:
            shard = bisect.bisect_right(self.starts, cursor) - 1
            parquet = self.files[shard]
            local = cursor - self.starts[shard]
            group_start = 0
            for group in range(parquet.num_row_groups):
                group_length = parquet.metadata.row_group(group).num_rows
                if local < group_start + group_length:
                    break
                group_start += group_length
            cache_key = (shard, group)
            if cache_key not in self._row_groups:
                # One contiguous row-group range avoids a separate redirected
                # HTTP request per column. PyArrow's exact column reads are
                # then served from this bounded in-memory compressed cache.
                metadata = parquet.metadata.row_group(group)
                chunks = [metadata.column(i) for i in range(metadata.num_columns)
                          if self.columns is None or
                          metadata.column(i).path_in_schema.split(".")[0] in self.columns]
                starts = [column.dictionary_page_offset or column.data_page_offset
                          for column in chunks]
                start = min(starts)
                stop = max(offset + column.total_compressed_size
                           for offset, column in zip(starts, chunks))
                self.readers[shard].seek(start)
                self.readers[shard].read(stop - start)
                self._row_groups[cache_key] = parquet.read_row_group(
                    group, columns=self.columns, use_threads=False)
                while len(self._row_groups) > 4:
                    self._row_groups.popitem(last=False)
            else:
                self._row_groups.move_to_end(cache_key)
            table = self._row_groups[cache_key]
            take = min(end - cursor, group_start + group_length - local)
            for row in table.slice(local - group_start, take).to_pylist():
                rows.append({"row_idx": cursor, "row": row})
                cursor += 1
        return {"num_rows_total": self.num_rows, "rows": rows}

    @property
    def bytes_downloaded(self):
        return sum(reader.bytes_downloaded for reader in self.readers)


def conversion_urls(dataset: str, config: str, split: str,
                    source_revision: str, *, session=None) -> tuple[list[str], str]:
    """Resolve an explicitly pinned conversion after checking the source pin.

    This checks both immutable identities, not a cryptographic conversion/source
    association. Before replacing another transport, compare representative row
    content and exact split length and retain both SHAs in run provenance.
    """
    session = session or requests.Session()
    conversion = PINNED_CONVERSIONS[dataset]
    base = f"https://huggingface.co/api/datasets/{dataset}"
    source = session.get(base, timeout=30)
    source.raise_for_status()
    if source.json().get("sha") != source_revision:
        raise ValueError("source revision changed; conversion association needs revalidation")
    response = session.get(f"{base}/revision/{conversion}", timeout=30)
    response.raise_for_status()
    metadata = response.json()
    if metadata.get("sha") != conversion:
        raise ValueError("conversion revision does not match the pinned commit")
    prefixes = (f"{config}/{split}/", f"{config}/partial-{split}/")
    paths = sorted(item["rfilename"] for item in metadata["siblings"]
                   if item["rfilename"].startswith(prefixes)
                   and item["rfilename"].endswith(".parquet"))
    if not paths:
        raise ValueError("no converted Parquet shards found for this split")
    return ([f"https://huggingface.co/datasets/{dataset}/resolve/{conversion}/"
             f"{quote(path, safe='/')}" for path in paths], conversion)


def parquet_page(dataset: str, config: str, split: str, source_revision: str,
                 offset: int, length: int) -> dict:
    """Rows-API-compatible page from a cached immutable Parquet reader.

    All columns are read so raw cache rows and their canonical hashes match
    the ordinary API transport. The RAID conversion, like its rows API view,
    is partial; the exact converted row count is checked before sampling.
    """
    key = (dataset, config, split, source_revision)
    if key not in _READERS:
        urls, conversion = conversion_urls(dataset, config, split, source_revision)
        _READERS[key] = (ParquetWindows(urls, expected_rows=SPLIT_ROWS[dataset]), conversion)
    reader, conversion = _READERS[key]
    return {**reader.get_rows(offset, length), "partial": dataset == "liamdugan/raid",
            "source_revision": source_revision, "conversion_revision": conversion,
            "transport": "immutable_parquet_ranges"}
