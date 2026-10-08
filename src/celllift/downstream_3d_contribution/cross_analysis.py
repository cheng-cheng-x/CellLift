from celllift.runtime import resource_path as _public_resource
from .common import *
from .phenotype import analyze
import pandas as pd
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', action='store_true')
    args = p.parse_args()
    for ds in ('bracs', 'tcga_crc_msi', 'tcga_brca'):
        units = pd.read_parquet(OUT / ds / 'shared/units.parquet')
        for task in TASKS[ds]:
            root = OUT / ds / task
            records = []
            for split in ('train', 'fit', 'test'):
                for path in (root / 'model/predict' / split).glob('*.json'):
                    r = read(path)
                    if r.get('status') != 'complete':
                        continue
                    row = dict(sample_id=r['sample_id'], split='train' if split == 'fit' else split)
                    for k, p in enumerate(r['probability2'][1:]):
                        row[f'probability2_{k + 1}'] = p
                    records.append(row)
            dest = root / 'synthesis' / ('conditional_on_2d_and_image' if args.image else 'conditional_on_2d_model')
            if not records:
                write(dest / 'status.json', dict(status='missing_model_predictions'))
                continue
            scores = pd.DataFrame(records).drop_duplicates(['sample_id', 'split'])
            cols = [c for c in scores if c.startswith('probability2_')]
            if args.image:
                file = root / 'model/background/image_baseline/image_probabilities.csv'
                if not file.exists():
                    write(dest / 'status.json', dict(status='missing_image_background_predictions'))
                    continue
                image = pd.read_csv(file).drop_duplicates(['sample_id', 'split'])
                scores = scores.merge(image, on=['sample_id', 'split'], how='inner')
                cols += [c for c in image if c.startswith('image_probability_')]
            t = units[units.task == task].merge(scores, on=['sample_id', 'split'], how='inner')
            ref = t[t.split == 'train']
            test = t[t.split == 'test']
            if ref.empty or test.empty:
                write(dest / 'status.json', dict(status='incomplete_predictions', development=len(ref), test=len(test)))
                continue
            analyze(test, ref, dest / 'test', LABELS[task], score_columns=cols)
            write(dest / 'status.json', dict(status='complete_for_available_predictions', development=len(ref), test=len(test), expected_test=int(((units.task == task) & (units.split == 'test')).sum()), interpretation='additional conditional association; not a requirement for primary phenotype results'))
if __name__ == '__main__':
    main()
