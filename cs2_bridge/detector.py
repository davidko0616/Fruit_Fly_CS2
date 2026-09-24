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
        summary = json.loads((self.root / 'summary.json').read_text(encoding='utf-8'))
        self.class_names = tuple(summary['class_names'])
        if not self.class_names:
            raise ValueError('Detector dataset must define at least one class')
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
        labels = torch.tensor([
            int(box['class_id']) + 1 for box in row['boxes']
        ], dtype=torch.int64)
        if len(labels) and (int(labels.min()) < 1 or
                            int(labels.max()) > len(self.class_names)):
            raise ValueError('Box class ID is outside dataset class names')
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([index], dtype=torch.int64),
            'area': ((boxes[:, 2] - boxes[:, 0]) *
                     (boxes[:, 3] - boxes[:, 1])),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image, target, row


def collate_detection_batch(batch):
    images, targets, metadata = zip(*batch)
    return list(images), list(targets), list(metadata)


def build_player_ssdlite(pretrained=True, image_size=320, foreground_classes=1):
    """Create a player detector, retaining COCO person initialization per class."""
    if image_size <= 0 or image_size % 32:
        raise ValueError('image_size must be a positive multiple of 32')
    if foreground_classes <= 0:
        raise ValueError('foreground_classes must be positive')
    weights = (SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
               if pretrained else None)
    model = ssdlite320_mobilenet_v3_large(
        weights=weights, weights_backbone=None)
    old_head = model.head.classification_head
    channels = [block[1].in_channels for block in old_head.module_list]
    anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = lambda channels: torch.nn.BatchNorm2d(
        channels, eps=0.001, momentum=0.03)
    output_classes = foreground_classes + 1
    new_head = SSDLiteClassificationHead(
        channels, anchors, output_classes, norm_layer)

    with torch.no_grad():
        for old_block, new_block, anchor_count in zip(
                old_head.module_list, new_head.module_list, anchors):
            new_block[0].load_state_dict(old_block[0].state_dict())
            old_conv, new_conv = old_block[1], new_block[1]
            old_classes = old_conv.out_channels // anchor_count
            for anchor in range(anchor_count):
                old_indices = torch.tensor(
                    [anchor * old_classes] +
                    [anchor * old_classes + 1] * foreground_classes)
                new_start = anchor * output_classes
                new_conv.weight[new_start:new_start + output_classes].copy_(
                    old_conv.weight[old_indices])
                new_conv.bias[new_start:new_start + output_classes].copy_(
                    old_conv.bias[old_indices])
    model.head.classification_head = new_head
    model.transform.min_size = (image_size,)
    model.transform.max_size = image_size
    model.transform.fixed_size = (image_size, image_size)
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


def evaluate_detection_records(records, score_threshold=0.25, iou_threshold=0.5,
                               class_names=None):
    """Compute fixed-threshold person metrics and grouped ground-truth recall."""
    totals = Counter(tp=0, fp=0, fn=0, negative_frames=0,
                     false_positives_on_negative_frames=0,
                     class_correct=0)
    group_totals = defaultdict(lambda: Counter(gt=0, matched=0))
    class_totals = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    confusion = Counter()

    for record in records:
        ground_truth = torch.as_tensor(record['ground_truth'], dtype=torch.float32).reshape(-1, 4)
        predicted = torch.as_tensor(record['predicted_boxes'], dtype=torch.float32).reshape(-1, 4)
        scores = torch.as_tensor(record['scores'], dtype=torch.float32)
        predicted_labels = torch.as_tensor(
            record.get('predicted_labels', [1] * len(predicted)), dtype=torch.int64)
        keep = scores >= score_threshold
        predicted, scores = predicted[keep], scores[keep]
        predicted_labels = predicted_labels[keep]
        order = torch.argsort(scores, descending=True)
        predicted = predicted[order]
        predicted_labels = predicted_labels[order]
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
                matches.append((prediction_index, index))

        true_positives = len(matches)
        false_positives = len(predicted) - true_positives
        false_negatives = len(ground_truth) - true_positives
        totals.update(tp=true_positives, fp=false_positives, fn=false_negatives)
        if not len(ground_truth):
            totals['negative_frames'] += 1
            totals['false_positives_on_negative_frames'] += false_positives

        matched = {ground_truth_index for _, ground_truth_index in matches}
        boxes_metadata = record['boxes_metadata']
        for prediction_index, ground_truth_index in matches:
            truth_label = int(boxes_metadata[ground_truth_index].get('class_id', 0)) + 1
            predicted_label = int(predicted_labels[prediction_index])
            if predicted_label == truth_label:
                totals['class_correct'] += 1
            truth_name = (class_names[truth_label - 1]
                          if class_names and truth_label <= len(class_names)
                          else str(truth_label))
            predicted_name = (class_names[predicted_label - 1]
                              if class_names and predicted_label <= len(class_names)
                              else str(predicted_label))
            confusion[f'{truth_name}->{predicted_name}'] += 1

        truth_labels = torch.tensor([
            int(box.get('class_id', 0)) + 1 for box in boxes_metadata
        ], dtype=torch.int64)
        label_ids = set(predicted_labels.tolist()) | set(truth_labels.tolist())
        for label_id in label_ids:
            class_predictions = torch.where(predicted_labels == label_id)[0]
            class_truth = torch.where(truth_labels == label_id)[0]
            class_ious = ious[class_predictions][:, class_truth]
            used_truth = set()
            class_matches = 0
            for prediction_row in range(len(class_predictions)):
                if not len(class_truth):
                    break
                candidates = class_ious[prediction_row].clone()
                if used_truth:
                    candidates[list(used_truth)] = -1
                value, truth_row = candidates.max(dim=0)
                if float(value) >= iou_threshold:
                    used_truth.add(int(truth_row))
                    class_matches += 1
            counts = class_totals[int(label_id)]
            counts['tp'] += class_matches
            counts['fp'] += len(class_predictions) - class_matches
            counts['fn'] += len(class_truth) - class_matches
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
    per_class = {}
    for label_id, values in sorted(class_totals.items()):
        name = (class_names[label_id - 1]
                if class_names and 0 < label_id <= len(class_names)
                else str(label_id))
        class_tp, class_fp, class_fn = values['tp'], values['fp'], values['fn']
        per_class[name] = {
            'true_positives': class_tp,
            'false_positives': class_fp,
            'false_negatives': class_fn,
            'precision': class_tp / (class_tp + class_fp)
            if class_tp + class_fp else 0.0,
            'recall': class_tp / (class_tp + class_fn)
            if class_tp + class_fn else 0.0,
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
        'per_class': per_class,
        'matched_classification': {
            'correct': totals['class_correct'],
            'total': tp,
            'accuracy': totals['class_correct'] / tp if tp else 0.0,
            'confusion': dict(sorted(confusion.items())),
        },
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
                'source_frame_id': row['source_frame_id'],
                'image': row['image'],
                'ground_truth': target['boxes'].tolist(),
                'boxes_metadata': row['boxes'],
                'predicted_boxes': prediction['boxes'].cpu().tolist(),
                'scores': prediction['scores'].cpu().tolist(),
                'predicted_labels': prediction['labels'].cpu().tolist(),
            })
    return records, latencies


def evaluate_model(model, data_loader, device, score_threshold=0.25,
                   iou_threshold=0.5, class_names=None):
    records, latencies = collect_detection_records(model, data_loader, device)
    metrics = evaluate_detection_records(
        records, score_threshold, iou_threshold, class_names)
    metrics['latency_ms_mean'] = 1000 * sum(latencies) / len(latencies)
    metrics['latency_ms_median'] = 1000 * median(latencies)
    metrics['evaluated_frames'] = len(records)
    return metrics
