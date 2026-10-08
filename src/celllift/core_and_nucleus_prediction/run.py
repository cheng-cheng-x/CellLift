from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import os
from .evaluate import recommended_lizard_b, write_tables
from .fields import cache_c3_fields, cache_dino_spatial
from .fuse import fuse_pair
from .official_fcn import ensure_repo, run_seed42_fcn, verify_released_fcn
from .predict import predict_all, predict_arm
from .probe import fit_probe
from .train import ARMS, train_arm

def fusion_pairs() -> list[tuple[str, str, str]]:
    rec = recommended_lizard_b()
    return [('arvaniti', 'B_paper', 'G2'), ('arvaniti', 'B_paper', 'G3'), ('arvaniti', 'B_paper', 'GR'), ('arvaniti', 'B_paper', 'G23'), ('arvaniti', 'B_paper', 'G2R'), ('lizard', rec, 'N2'), ('lizard', rec, 'N3'), ('lizard', rec, 'NR'), ('lizard', rec, 'N23'), ('lizard', rec, 'N2R'), ('lizard', rec, 'J2'), ('lizard', rec, 'JS'), ('lizard', rec, 'JR')]

def main() -> None:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('clone-author', 'verify-fcn', 'seed42-fcn', 'fcn-diagnose', 'probe', 'train', 'predict', 'spatial', 'fields', 'fuse', 'tables', 'rescore'))
    parser.add_argument('--arm', default='G2')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dataset', default='arvaniti')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    args = parser.parse_args()
    if args.stage == 'clone-author':
        print(json.dumps({'repo': str(ensure_repo())}, indent=2))
        return
    if args.stage == 'verify-fcn':
        print(json.dumps(verify_released_fcn(args.device), indent=2, default=str))
        return
    if args.stage == 'fcn-diagnose':
        from .official_fcn import diagnose_fcn_permutations
        print(json.dumps(diagnose_fcn_permutations(), indent=2, default=str))
        return
    if args.stage == 'seed42-fcn':
        print(json.dumps(run_seed42_fcn(args.device), indent=2, default=str))
        return
    if args.stage == 'probe':
        print(json.dumps(fit_probe(args.dataset, args.device), indent=2, default=str))
        return
    if args.stage == 'train':
        print(json.dumps(train_arm(args.arm, args.device), indent=2, default=str))
        return
    if args.stage == 'predict':
        if args.arm == 'all':
            print(json.dumps(predict_all(args.device), indent=2, default=str)[:8000])
        else:
            print(json.dumps(predict_arm(args.arm, args.device), indent=2, default=str)[:8000])
        return
    if args.stage == 'rescore':
        from .predict import rescore_arvaniti_arm
        from .worker import ORDER
        arms = [a for a in ORDER if not a.startswith(('N', 'J', 'LH', 'LA')) and a not in {'B_crop', 'B_dino'}] if args.arm == 'all' else [args.arm]
        print(json.dumps({arm: rescore_arvaniti_arm(arm) for arm in arms}, indent=2, default=str)[:12000])
        return
    if args.stage == 'spatial':
        print(json.dumps(cache_dino_spatial(args.device), indent=2, default=str))
        return
    if args.stage == 'fields':
        print(json.dumps(cache_c3_fields(shard=args.shard, n_shards=args.n_shards), indent=2, default=str))
        return
    if args.stage == 'fuse':
        print(json.dumps([fuse_pair(dataset, left, right) for dataset, left, right in fusion_pairs()], indent=2))
        return
    if args.stage == 'tables':
        print(json.dumps(write_tables(), indent=2, default=str)[:8000])
if __name__ == '__main__':
    main()
