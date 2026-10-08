from celllift.runtime import resource_path as _public_resource
from .common import *
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
_MATCHERS = {}

class Matcher:

    def __new__(cls, adapter, image=False):
        key = (id(adapter), bool(image))
        if key not in _MATCHERS:
            _MATCHERS[key] = super().__new__(cls)
        return _MATCHERS[key]

    def __init__(self, adapter, image=False):
        if getattr(self, '_initialized', False):
            return
        self.a = adapter
        self.image = image
        folder = OUT / adapter.dataset / adapter.task / 'model/shared'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ('donors_fit_only_image.npz' if image else 'donors_fit_only.npz')
        if path.exists():
            with np.load(path) as z:
                pool = {k: z[k] for k in z.files}
        else:
            values = []
            rich = []
            clusters = []
            images = []
            for i, r in enumerate(adapter.rows):
                if r['split'] not in ('train', 'fit'):
                    continue
                if hasattr(adapter, 'training_ids') and str(r['sample_id']) not in adapter.training_ids:
                    continue
                if 'graph_ids' in r and (not r['graph_ids']):
                    continue
                d = adapter.input(r)
                for gid in np.unique(d['groups']):
                    ix = np.flatnonzero((d['groups'] == gid) & d['valid'])
                    rng = np.random.default_rng(stable(r['sample_id']) + int(gid))
                    ix = rng.choice(ix, min(64, len(ix)), replace=False)
                    if not len(ix):
                        continue
                    values.append(self.raw(d)[ix])
                    rich.append(d['rich'][ix])
                    clusters.extend([r['cluster_id']] * len(ix))
                    if not image:
                        images.append(d['image'][ix])
                if i % 100 == 0:
                    print('donor units', i, flush=True)
            pool = dict(raw=np.concatenate(values), rich=np.concatenate(rich), clusters=np.asarray(clusters, dtype=str))
            np.savez_compressed(path, **pool)
            if not image:
                image_pool = dict(pool, raw=np.concatenate([pool['raw'], np.concatenate(images)], axis=1))
                np.savez_compressed(folder / 'donors_fit_only_image.npz', **image_pool)
                del image_pool, images
        self.pool = pool
        self.mean = pool['raw'].mean(0)
        self.std = pool['raw'].std(0)
        self.std[self.std < 1e-08] = 1
        self.x = (pool['raw'] - self.mean) / self.std
        self.pca = PCA(n_components=min(8, self.x.shape[1]), random_state=42).fit(self.x)
        self.nn = NearestNeighbors(n_neighbors=min(64, len(self.x)), algorithm='kd_tree', n_jobs=4).fit(self.pca.transform(self.x))
        if adapter.dataset == 'tcga_crc_msi':
            from .exact_matching import ExactTree
            self.nn = ExactTree(self.nn)
        fingerprint = hashlib.sha256()
        fingerprint.update(digest(path).encode())
        for value in (self.mean, self.std, self.pca.components_, self.pca.mean_):
            fingerprint.update(np.ascontiguousarray(value).tobytes())
        self.query_cache = folder / 'query_cache' / fingerprint.hexdigest()
        threshold = folder / ('threshold_fit_only_image.json' if image else 'threshold_fit_only.json')
        if threshold.exists():
            self.limit = read(threshold)['distance_limit']
        else:
            distances = []
            for number, r in enumerate(adapter.rows):
                if r['split'] not in ('train', 'fit'):
                    continue
                if 'graph_ids' in r and (not r['graph_ids']):
                    continue
                d = adapter.calibration_input(r, image=image) if hasattr(adapter, 'calibration_input') else adapter.input(r)
                ix = np.flatnonzero(d['valid'])
                ix = ix[np.argsort([stable(str(r['sample_id']) + ':' + str(j)) for j in ix])[:64]]
                _, dist = self.query(self.raw(d)[ix], r['cluster_id'])
                distances.extend(dist[:, 0][np.isfinite(dist[:, 0])])
                if number % 100 == 0:
                    print('development caliper', number, 'image', image, 'queries', len(distances), flush=True)
            self.limit = float(np.quantile(distances, 0.95)) if distances else float('nan')
            write(threshold, dict(distance_limit=self.limit, quantile=0.95, development_queries=len(distances), exclude_same_cluster=True, pca_candidates=64, final_candidates=5, max_nodes_per_training_graph=64))
        self._initialized = True

    def raw(self, d):
        return np.concatenate([d['two'], d['image']], 1) if self.image else d['two']

    def query(self, raw, cluster):
        if not len(raw):
            return (np.empty((0, 5), int), np.empty((0, 5)))
        h = hashlib.sha256(str((str(cluster), raw.shape, raw.dtype.str)).encode())
        h.update(memoryview(np.ascontiguousarray(raw)).cast('B'))
        path = self.query_cache / (h.hexdigest() + '.npz')
        if path.exists():
            with np.load(path) as z:
                return (z['indices'], z['distances'])
        from .exact_matching import query
        indices, distances = query(raw, cluster, self.mean, self.std, self.pca, self.nn, self.x, self.pool['clusters'])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp.npz')
        np.savez(tmp, indices=indices, distances=distances)
        tmp.replace(path)
        return (indices, distances)

    def references(self, d, cluster, repeats=20):
        idx, dist = self.query(self.raw(d), cluster)
        ok = (dist <= self.limit) & d['valid'][:, None]
        out = []
        coverage = []
        for seed in range(42, 42 + repeats):
            rng = np.random.default_rng(seed)
            score = rng.random(ok.shape)
            score[~ok] = -1
            pick = score.argmax(1)
            eligible = ok.any(1)
            x = d['rich'].copy()
            x[eligible] = self.pool['rich'][idx[np.arange(len(idx)), pick]][eligible]
            out.append(x)
            coverage.append(eligible)
        return (np.stack(out), np.stack(coverage))

    def permutations(self, d, repeats=20):
        x = (self.raw(d) - self.mean) / self.std
        results = []
        coverage = []
        candidates = {}
        blocks = []
        for g in np.unique(d['groups']):
            ids = np.flatnonzero((d['groups'] == g) & d['valid'])
            blocks.append(ids)
            for start in range(0, len(ids), 512):
                source = ids[start:start + 512]
                distance = cdist(x[source], x[ids]) / np.sqrt(x.shape[1])
                for u, dd in zip(source, distance):
                    candidates[int(u)] = ids[(dd <= self.limit) & (ids != u)]
        for seed in range(42, 42 + repeats):
            rng = np.random.default_rng(seed)
            order = np.arange(len(x))
            changed = np.zeros(len(x), bool)
            pending = d['valid'].copy()
            for ids in blocks:
                for u in rng.permutation(ids):
                    if not pending[u]:
                        continue
                    pending[u] = False
                    v = candidates[int(u)]
                    c = v[pending[v]]
                    if len(c):
                        j = int(rng.choice(c))
                        pending[j] = False
                        order[u] = j
                        order[j] = u
                        changed[[u, j]] = True
            results.append(d['rich'][order])
            coverage.append(changed)
        return (np.stack(results), np.stack(coverage))
