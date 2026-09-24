"""Evaluate a saved visible-player detector at fixed validation/test thresholds."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from cs2_bridge.detector import (
    VisiblePlayerDataset,
    build_player_ssdlite,
    collect_detection_records,
    collate_detection_batch,
    evaluate_detection_records,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'),
                        default='validation')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.25])
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--nms-threshold', type=float, default=0.55)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--cpu-threads', type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if any(not 0 <= threshold <= 1 for threshold in args.thresholds):
        parser.error('score thresholds must be in [0, 1]')
    if args.output and args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')

    torch.set_num_threads(args.cpu_threads)
    device = torch.device('cpu')
    dataset = VisiblePlayerDataset(args.dataset, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_detection_batch)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = int(checkpoint.get('config', {}).get('image_size', 320))
    class_names = tuple(checkpoint.get('classes', ('background', 'player'))[1:])
    model = build_player_ssdlite(
        pretrained=False, image_size=image_size,
        foreground_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.nms_thresh = args.nms_threshold
    records, latencies = collect_detection_records(model, loader, device)
    results = []
    for threshold in args.thresholds:
        metrics = evaluate_detection_records(
            records, threshold, args.iou_threshold, class_names)
        metrics['evaluated_frames'] = len(records)
        metrics['latency_ms_mean'] = 1000 * sum(latencies) / len(latencies)
        results.append(metrics)
    result = {
        'schema_version': 1,
        'checkpoint': str(args.checkpoint),
        'checkpoint_epoch': checkpoint['epoch'],
        'image_size': image_size,
        'class_names': list(class_names),
        'split': args.split,
        'nms_threshold': args.nms_threshold,
        'metrics': results,
    }
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')


if __name__ == '__main__':
    main()
