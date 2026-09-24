"""Train and validate the first CPU visible-player detector pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

from cs2_bridge.detector import (
    VisiblePlayerDataset,
    build_player_ssdlite,
    collate_detection_batch,
    evaluate_model,
)


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _save_checkpoint(path, model, optimizer, epoch, config, metrics):
    torch.save({
        'schema_version': 1,
        'architecture': 'ssdlite320_mobilenet_v3_large',
        'classes': ['background', 'player'],
        'epoch': epoch,
        'config': config,
        'validation_metrics': metrics,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def train(args):
    dataset_root = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    output.mkdir(parents=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device('cpu')

    train_dataset = VisiblePlayerDataset(dataset_root, 'train')
    validation_dataset = VisiblePlayerDataset(dataset_root, 'validation')
    if len(train_dataset) < args.batch_size:
        raise ValueError('Batch size exceeds the number of training frames')
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_detection_batch, generator=generator,
        drop_last=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_detection_batch)

    model = build_player_ssdlite(pretrained=not args.no_pretrained).to(device)
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    config = {
        'dataset': str(dataset_root),
        'dataset_summary_sha256': _sha256(dataset_root / 'summary.json'),
        'seed': args.seed, 'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'freeze_backbone': args.freeze_backbone,
        'pretrained_coco_person_initialization': not args.no_pretrained,
        'score_threshold': args.score_threshold,
        'iou_threshold': args.iou_threshold,
        'cpu_threads': args.cpu_threads,
    }
    (output / 'config.json').write_text(
        json.dumps(config, indent=2) + '\n', encoding='utf-8')
    environment = {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'device': str(device),
    }
    (output / 'environment.json').write_text(
        json.dumps(environment, indent=2) + '\n', encoding='utf-8')

    history, best_key = [], (-1.0, -1.0)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
        losses = []
        epoch_started = time.perf_counter()
        for images, targets, _ in train_loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()}
                       for target in targets]
            optimizer.zero_grad(set_to_none=True)
            loss_parts = model(images, targets)
            loss = sum(loss_parts.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        metrics = evaluate_model(
            model, validation_loader, device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold)
        row = {
            'epoch': epoch,
            'mean_training_loss': sum(losses) / len(losses),
            'training_batches': len(losses),
            'epoch_seconds': time.perf_counter() - epoch_started,
            'validation': metrics,
        }
        history.append(row)
        (output / 'history.json').write_text(
            json.dumps(history, indent=2) + '\n', encoding='utf-8')
        _save_checkpoint(output / 'last.pt', model, optimizer, epoch,
                         config, metrics)
        key = (metrics['f1'], metrics['recall'])
        if key > best_key:
            best_key = key
            _save_checkpoint(output / 'best.pt', model, optimizer, epoch,
                             config, metrics)
        print(json.dumps({
            'epoch': epoch, 'loss': row['mean_training_loss'],
            'validation_f1': metrics['f1'],
            'validation_precision': metrics['precision'],
            'validation_recall': metrics['recall'],
            'seconds': row['epoch_seconds'],
        }), flush=True)

    summary = {
        'status': 'complete',
        'epochs': args.epochs,
        'training_seconds': time.perf_counter() - started,
        'best_epoch': max(history, key=lambda row: (
            row['validation']['f1'], row['validation']['recall']))['epoch'],
        'best_validation': max(history, key=lambda row: (
            row['validation']['f1'], row['validation']['recall']))['validation'],
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu-threads', type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument('--score-threshold', type=float, default=0.25)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--freeze-backbone', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.cpu_threads <= 0:
        parser.error('epochs, batch size, and CPU threads must be positive')
    print(json.dumps(train(args), indent=2))


if __name__ == '__main__':
    main()
