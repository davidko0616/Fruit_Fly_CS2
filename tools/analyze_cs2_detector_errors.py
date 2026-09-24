"""List per-box detector localization evidence for validation error analysis."""
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
    box_iou,
    build_player_ssdlite,
    collect_detection_records,
    collate_detection_batch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'),
                        default='validation')
    parser.add_argument('--score-threshold', type=float, default=0.25)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--nms-threshold', type=float, default=0.3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--cpu-threads', type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    dataset = VisiblePlayerDataset(args.dataset, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_detection_batch)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    image_size = int(checkpoint.get('config', {}).get('image_size', 320))
    model = build_player_ssdlite(pretrained=False, image_size=image_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.nms_thresh = args.nms_threshold
    records, _ = collect_detection_records(model, loader, torch.device('cpu'))

    rows = []
    for record in records:
        predicted = torch.tensor(record['predicted_boxes'], dtype=torch.float32).reshape(-1, 4)
        scores = torch.tensor(record['scores'], dtype=torch.float32)
        keep = scores >= args.score_threshold
        predicted, scores = predicted[keep], scores[keep]
        truth = torch.tensor(record['ground_truth'], dtype=torch.float32).reshape(-1, 4)
        ious = box_iou(predicted, truth)
        boxes = []
        for index, metadata in enumerate(record['boxes_metadata']):
            if len(predicted):
                best_iou, prediction_index = ious[:, index].max(dim=0)
                best_iou = float(best_iou)
                best_score = float(scores[int(prediction_index)])
                best_prediction_box = predicted[int(prediction_index)].tolist()
            else:
                best_iou = best_score = 0.0
                best_prediction_box = None
            box = truth[index]
            boxes.append({
                'team': metadata['team'],
                'visibility': metadata['visibility'],
                'width': float(box[2] - box[0]),
                'height': float(box[3] - box[1]),
                'best_iou': best_iou,
                'best_score': best_score,
                'best_prediction_box': best_prediction_box,
                'matched': best_iou >= args.iou_threshold,
            })
        rows.append({
            'source_frame_id': record['source_frame_id'],
            'image': record['image'],
            'predictions_above_threshold': len(predicted),
            'ground_truth': boxes,
        })
    print(json.dumps({
        'checkpoint_epoch': checkpoint['epoch'],
        'image_size': image_size,
        'frames': rows,
    }, indent=2))


if __name__ == '__main__':
    main()
