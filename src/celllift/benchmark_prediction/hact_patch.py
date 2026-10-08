from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
HACT_CODE = Path(_resource_path('artifact_0001'))
HACT_OUT = Path(_resource_path('artifact_0004'))
SEED_BLOCK = '\n    import random\n    random.seed(42)\n    np.random.seed(42)\n    torch.manual_seed(42)\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(42)\n'
DUMP_BLOCK = f'\n        pred_dir = r"{HACT_OUT}"\n        os.makedirs(pred_dir, exist_ok=True)\n        dump = {{\n            "ckpt": metric,\n            "labels": [int(v) for v in all_test_labels],\n            "preds": [int(v) for v in all_test_preds],\n            "n": int(len(all_test_labels)),\n        }}\n        with open(os.path.join(pred_dir, "test_{{}}.json".format(metric)), "w", encoding="utf-8") as handle:\n            import json as _json\n            _json.dump(dump, handle)\n'

def run() -> dict:
    generate = HACT_CODE / 'core' / 'generate_hact_graphs.py'
    train = HACT_CODE / 'core' / 'train.py'
    report = {'generate': False, 'train': False}
    if generate.is_file():
        text = generate.read_text(encoding='utf-8')
        text = text.replace('MIN_NR_PIXELS = 50000', 'MIN_NR_PIXELS = 1')
        generate.write_text(text, encoding='utf-8')
        report['generate'] = True
    if train.is_file():
        text = train.read_text(encoding='utf-8')
        text = text.replace('config = yaml.load(f)', 'config = yaml.safe_load(f)')
        if 'random.seed(42)' not in text:
            text = text.replace('def main(args):', 'def main(args):' + SEED_BLOCK, 1)
        if 'test_{}.json'.format('best_val_weighted_f1_score') not in text and 'test_{}.json".format(metric)' not in text:
            marker = "print('Test weighted F1 score {}'.format(weighted_f1_score))"
            if marker in text:
                text = text.replace(marker, DUMP_BLOCK + '\n        ' + marker, 1)
        train.write_text(text, encoding='utf-8')
        report['train'] = True
    return report
if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
