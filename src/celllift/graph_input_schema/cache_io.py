from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
import csv
import os
from celllift.runtime import ResourcePath as Path
from typing import Iterator

def stable_shard(key: str, count: int) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), 'big') % count

def _sha256(path: Path, chunk_size: int=8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_lmdb_manifest(root: str | Path, prefixes: list[str], shards: int, output_name: str) -> list[dict[str, object]]:
    import lmdb
    root = Path(root)
    rows: list[dict[str, object]] = []
    for prefix in prefixes:
        for shard in range(int(shards)):
            directory = root / f'{prefix}_{shard:02d}.lmdb'
            data = directory / 'data.mdb'
            if not data.is_file():
                raise FileNotFoundError(data)
            env = lmdb.open(str(directory), subdir=True, readonly=True, lock=False, readahead=False)
            try:
                entries = int(env.stat()['entries'])
            finally:
                env.close()
            rows.append({'prefix': prefix, 'shard': shard, 'path': str(data), 'size_bytes': data.stat().st_size, 'entries': entries, 'sha256': _sha256(data)})
    destination = root / output_name
    temporary = destination.with_suffix(destination.suffix + f'.tmp.{os.getpid()}')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['prefix', 'shard', 'path', 'size_bytes', 'entries', 'sha256'], delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return rows

class ShardedLMDBWriter:

    def __init__(self, root: str | Path, prefix: str, shards: int, map_size: int):
        import lmdb
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shards = int(shards)
        self.envs = [lmdb.open(str(self.root / f'{prefix}_{i:02d}.lmdb'), subdir=True, map_size=map_size, max_dbs=1, lock=True, sync=False, metasync=False) for i in range(self.shards)]

    def put(self, key: str, value: bytes) -> int:
        shard = stable_shard(key, self.shards)
        env = self.envs[shard]
        while True:
            try:
                with env.begin(write=True) as txn:
                    if not txn.put(key.encode(), value, overwrite=False):
                        old = txn.get(key.encode())
                        if old != value:
                            raise RuntimeError(f'cache key already exists with different payload: {key}')
                return shard
            except __import__('lmdb').MapFullError:
                env.set_mapsize(env.info()['map_size'] * 2)

    def put_many(self, items: list[tuple[str, bytes]]) -> list[int]:
        grouped: dict[int, list[tuple[bytes, bytes]]] = {}
        result: list[int] = []
        for key, value in items:
            shard = stable_shard(key, self.shards)
            result.append(shard)
            grouped.setdefault(shard, []).append((key.encode(), value))
        for shard, values in grouped.items():
            env = self.envs[shard]
            while True:
                try:
                    with env.begin(write=True) as txn:
                        for key, value in values:
                            if not txn.put(key, value, overwrite=False):
                                old = txn.get(key)
                                if old != value:
                                    raise RuntimeError(f'cache key already exists with different payload: {key.decode()}')
                    break
                except __import__('lmdb').MapFullError:
                    env.set_mapsize(env.info()['map_size'] * 2)
        return result

    def close(self) -> None:
        for env in self.envs:
            env.sync()
            env.close()

class ShardedLMDBReader:

    def __init__(self, root: str | Path, prefix: str, shards: int):
        self.root = Path(root)
        self.prefix = prefix
        self.shards = int(shards)
        self._envs = None

    def _open(self):
        if self._envs is None:
            import lmdb
            self._envs = [lmdb.open(str(self.root / f'{self.prefix}_{i:02d}.lmdb'), subdir=True, readonly=True, lock=False, readahead=False, max_readers=2048) for i in range(self.shards)]
        return self._envs

    def get(self, key: str) -> bytes:
        env = self._open()[stable_shard(key, self.shards)]
        with env.begin(buffers=False) as txn:
            value = txn.get(key.encode())
        if value is None:
            raise KeyError(key)
        return bytes(value)

    def __getstate__(self):
        return {'root': self.root, 'prefix': self.prefix, 'shards': self.shards, '_envs': None}

    def close(self):
        if self._envs:
            for env in self._envs:
                env.close()
            self._envs = None
