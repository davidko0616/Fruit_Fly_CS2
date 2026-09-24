"""CPU-friendly visible-player detector training and evaluation helpers."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median
import time

import torch
from torch.utils.data import Dataset
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image


class VisiblePlayerDataset(Dataset):
    """Read exported frames while preserving label metadata for grouped metrics."""

    def __init__(self, root, split):
        self.root = Path(root)
        metadata = self.root / 'metadata' / f'{split}.jsonl'
        self.rows = [json.loads(line) for line in metadata.read_text(
            encoding='utf-8').splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f'No {split} samples in {metadata}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['image']) as loaded:
            image = pil_to_tensor(loaded.convert('RGB')).float().div_(255)
        boxes = torch.tensor([
            [box['x1'], box['y1'], box['x2'], box['y2']]
            for box in row['boxes']
        ], dtype=torch.float32).reshape(-1, 4)
        target = {
            'boxes': boxes,
            'labels': torch.ones(len(boxes), dtype=torch.int64),
            'image_id': torch.tensor([index], dtype=torch.int64),
            'area': ((boxes[:, 2] - boxes[:, 0]) *
                     (boxes[:, 3] - boxes[:, 1])),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image, target, row


def collate_detection_batch(batch):
    images, targets, metadata = zip(*batch)
    return list(images), list(targets), list(metadata)


def build_player_ssdlite(pretrained=True):
    """Create background/player SSDlite, retaining COCO person initialization."""
    weights = (SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
               if pretrained else None)
    model = ssdlite320_mobilenet_v3_large(
        weights=weights, weights_backbone=None)
    old_head = model.head.classification_head
    channels = [block[1].in_channels for block in old_head.module_list]
    anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = lambda channels: torch.nn.BatchNorm2d(
        channels, eps=0.001, momentum=0.03)
    new_head = SSDLiteClassificationHead(channels, anchors, 2, norm_layer)

    with torch.no_grad():
        for old_block, new_block, anchor_count in zip(
                old_head.module_list, new_head.module_list, anchors):
            new_block[0].load_state_dict(old_block[0].state_dict())
            old_conv, new_conv = old_block[1], new_block[1]
            old_classes = old_conv.out_channels // anchor_count
            for anchor in range(anchor_count):
                old_indices = torch.tensor([
                    anchor * old_classes,
                    anchor * old_classes + 1,
                ])
                new_start = anchor * 2
                new_conv.weight[new_start:new_start + 2].copy_(
                    old_conv.weight[old_indices])
                new_conv.bias[new_start:new_start + 2].copy_(
                    old_conv.bias[old_indices])
    model.head.classification_head = new_head
    return model


def box_iou(boxes_a, boxes_b):
    if not len(boxes_a) or not len(boxes_b):
        return torch.zeros((len(boxes_a), len(boxes_b)), dtype=torch.float32)
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    size = (bottom_right - top_left).clamp(min=0)
    intersection = size[..., 0] * size[..., 1]
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) *
              (boxes_a[:, 3] - boxes_a[:, 1]))
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) *
              (boxes_b[:, 3] - boxes_b[:, 1]))
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp(min=1e-12)


def evaluate_detection_records(records, score_threshold=0.25, iou_threshold=0.5):
    """Compute fixed-threshold person metrics and grouped ground-truth recall."""
    totals = Counter(tp=0, fp=0, fn=0, negative_frames=0,
                     false_positives_on_negative_frames=0)
    group_totals = defaultdict(lambda: Counter(gt=0, matched=0))

    for record in records:
        ground_truth = torch.as_tensor(record['ground_truth'], dtype=torch.float32).reshape(-1, 4)
        predicted = torch.as_tensor(record['predicted_boxes'], dtype=torch.float32).reshape(-1, 4)
        scores = torch.as_tensor(record['scores'], dtype=torch.float32)
        keep = scores >= score_threshold
        predicted, scores = predicted[keep], scores[keep]
        order = torch.argsort(scores, descending=True)
        predicted = predicted[order]
        ious = box_iou(predicted, ground_truth)
        matched_gt = set()
        matches = []
        for prediction_index in range(len(predicted)):
            if not len(ground_truth):
                break
            candidates = ious[prediction_index].clone()
            if matched_gt:
                candidates[list(matched_gt)] = -1
            value, ground_truth_index = candidates.max(dim=0)
            if float(value) >= iou_threshold:
                index = int(ground_truth_index)
                matched_gt.add(index)
                matches.append(index)

        true_positives = len(matches)
        false_positives = len(predicted) - true_positives
        false_negatives = len(ground_truth) - true_positives
        totals.update(tp=true_positives, fp=false_positives, fn=false_negatives)
        if not len(ground_truth):
            totals['negative_frames'] += 1
            totals['false_positives_on_negative_frames'] += false_positives

        matched = set(matches)
        boxes_metadata = record['boxes_metadata']
        for index, box in enumerate(boxes_metadata):
            for group in (f"team:{box['team']}",
                          f"visibility:{box['visibility']}"):
                group_totals[group]['gt'] += 1
                if index in matched:
                    group_totals[group]['matched'] += 1

    tp, fp, fn = totals['tp'], totals['fp'], totals['fn']
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    grouped_recall = {
        group: {
            'ground_truth': values['gt'], 'matched': values['matched'],
            'recall': values['matched'] / values['gt'] if values['gt'] else 0.0,
        }
        for group, values in sorted(group_totals.items())
    }
    return {
        'score_threshold': score_threshold,
        'iou_threshold': iou_threshold,
        'true_positives': tp, 'false_positives': fp, 'false_negatives': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'negative_frames': totals['negative_frames'],
        'false_positives_on_negative_frames':
            totals['false_positives_on_negative_frames'],
        'false_positives_per_negative_frame': (
            totals['false_positives_on_negative_frames'] / totals['negative_frames']
            if totals['negative_frames'] else 0.0),
        'grouped_recall': grouped_recall,
    }


@torch.inference_mode()
def collect_detection_records(model, data_loader, device):
    model.eval()
    records, latencies = [], []
    for images, targets, metadata in data_loader:
        images = [image.to(device) for image in images]
        start = time.perf_counter()
        predictions = model(images)
        elapsed = time.perf_counter() - start
        latencies.extend([elapsed / len(images)] * len(images))
        for prediction, target, row in zip(predictions, targets, metadata):
            records.append({
                'ground_truth': target['boxes'].tolist(),
                'boxes_metadata': row['boxes'],
                'predicted_boxes': prediction['boxes'].cpu().tolist(),
                'scores': prediction['scores'].cpu().tolist(),
            })
    return records, latencies


def evaluate_model(model, data_loader, device, score_threshold=0.25,
                   iou_threshold=0.5):
    records, latencies = collect_detection_records(model, data_loader, device)
    metrics = evaluate_detection_records(records, score_threshold, iou_threshold)
    metrics['latency_ms_mean'] = 1000 * sum(latencies) / len(latencies)
    metrics['latency_ms_median'] = 1000 * median(latencies)
    metrics['evaluated_frames'] = len(records)
    return metrics
