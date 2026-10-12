import io, struct, zipfile, zlib, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HF = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main"

def make_session(total=10, backoff=1.5):
    """Session that transparently retries 429/5xx with exponential backoff,
    honouring Retry-After when HF sends it."""
    s = requests.Session()
    r = Retry(total=total, connect=total, read=total, status=total,
              backoff_factor=backoff,
              status_forcelist=[429, 500, 502, 503, 504],
              allowed_methods=["GET", "HEAD"],
              respect_retry_after_header=True,
              raise_on_status=False)
    ad = HTTPAdapter(max_retries=r, pool_connections=64, pool_maxsize=64)
    s.mount("https://", ad); s.mount("http://", ad)
    return s
SPLIT_PARTS = {73,76,78,83,88,89,92,95,96,102,103,111,118,183,184,185}

def part_urls(i):
    if i in SPLIT_PARTS:
        return [f"{HF}/OpenVid_part{i}_partaa", f"{HF}/OpenVid_part{i}_partab"]
    return [f"{HF}/OpenVid_part{i}.zip"]

class RangeFile(io.RawIOBase):
    """Several URLs concatenated into one seekable read-only file."""
    def __init__(self, urls, session=None, timeout=60):
        self.urls, self.s, self.timeout = list(urls), session or make_session(), timeout
        self.sizes = [int(self.s.head(u, allow_redirects=True, timeout=timeout)
                          .headers["Content-Length"]) for u in self.urls]
        self.offs, acc = [], 0
        for sz in self.sizes:
            self.offs.append(acc); acc += sz
        self.size, self.pos, self.bytes_fetched = acc, 0, 0

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def tell(self):     return self.pos
    def seekable(self): return True
    def readable(self): return True

    def read(self, n=-1):
        n = self.size - self.pos if (n is None or n < 0) else min(n, self.size - self.pos)
        if n <= 0: return b""
        out = bytearray()
        while n > 0:
            k = max(j for j, o in enumerate(self.offs) if o <= self.pos)
            local = self.pos - self.offs[k]
            take = min(n, self.sizes[k] - local)
            r = self.s.get(self.urls[k], headers={"Range": f"bytes={local}-{local+take-1}"},
                           allow_redirects=True, timeout=self.timeout)
            r.raise_for_status()
            out += r.content
            self.pos += len(r.content); n -= len(r.content); self.bytes_fetched += len(r.content)
            if len(r.content) < take: break
        return bytes(out)

def open_part(i, session=None):
    rf = RangeFile(part_urls(i), session or make_session())
    return zipfile.ZipFile(rf), rf


def read_member(z, rf, member):
    """Read one zip member with a SINGLE HTTP Range request.

    zipfile.ZipFile.read() costs 3-4 round trips per member (local header, file
    name, extra field, payload). Over HTTP that is latency- and request-rate-
    bound, and HF throttles on request count, so we fetch
    [local header .. end of payload] in one go and parse the 30-byte header
    locally. Members are deflated; zlib with a raw (-15) window undoes that.
    """
    zi = z.getinfo(member)
    if zi.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        return z.read(member)                       # exotic method: stay correct
    SLACK = 30 + 1024                               # header + filename + extra
    rf.seek(zi.header_offset)
    blob = rf.read(SLACK + zi.compress_size)
    if len(blob) < 30 or blob[:4] != b"PK\x03\x04":
        raise IOError(f"bad local header for {member}")
    fnl, exl = struct.unpack("<HH", blob[26:30])
    off = 30 + fnl + exl
    raw = blob[off:off + zi.compress_size]
    if len(raw) < zi.compress_size:                 # filename+extra > SLACK
        rf.seek(zi.header_offset + off + len(raw))
        raw += rf.read(zi.compress_size - len(raw))
    if len(raw) != zi.compress_size:
        raise IOError(f"short read {member}: {len(raw)}/{zi.compress_size}")
    data = raw if zi.compress_type == zipfile.ZIP_STORED else zlib.decompress(raw, -15)
    if len(data) != zi.file_size:
        raise IOError(f"bad size {member}: {len(data)}/{zi.file_size}")
    return data
