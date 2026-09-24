# Dust II visible-player detector pilot

## Scope

This is an intermediate CPU feasibility run for the screen-visible body detector.
It does not establish final accuracy and has not been evaluated on the held-out
test route. The detector receives only captured RGB frames; radar contacts and
game-state opponent coordinates are not inputs.

The accepted local dataset contains four training sessions and one independent
validation session:

- train: 75 frames, 44 positive frames, 31 verified negatives, 51 boxes;
- validation: 20 frames, nine positive frames, 11 verified negatives, 11 boxes.

Capture sessions remain intact within one split. Full-resolution images and
checkpoints remain local under `artifacts/cs2_bridge/`.

## Fixed training run

- model: TorchVision SSDlite320 MobileNetV3 Large;
- initialization: official COCO background/person weights;
- runtime: PyTorch 2.7.1+cpu and TorchVision 0.22.1+cpu;
- seed: 42;
- optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`;
- batch size: two, 15 epochs, frozen MobileNet backbone;
- checkpoint selection: validation F1 at score 0.25 and IoU 0.50;
- selected checkpoint: epoch six.

The official PyTorch compatibility table pairs PyTorch 2.7.1 with TorchVision
0.22.1. TorchVision describes SSDlite320 as a compact detector intended for
efficient inference:

- <https://pytorch.org/get-started/previous-versions/>
- <https://pytorch.org/blog/torchvision-ssdlite-implementation/>

## Validation calibration and result

NMS and score calibration used only the validation split. The frozen pilot
settings, selected before any test capture, are:

- score threshold: 0.25;
- NMS threshold: 0.30;
- match IoU: 0.50.

At those settings the selected checkpoint reports:

| Metric | Result |
|---|---:|
| Precision | 40.0% |
| Recall | 54.5% |
| F1 | 0.462 |
| Enemy recall | 83.3% (5/6) |
| Friendly recall | 20.0% (1/5) |
| Full-body recall | 71.4% (5/7) |
| Partial-body recall | 25.0% (1/4) |
| False positives on negative frames | 0 across 11 frames |
| Mean CPU latency | about 40 ms/frame |

Lowering the score threshold to 0.20 raises no additional matched bodies at NMS
0.30 but produces 22 false positives. Raising it above 0.25 loses recall. The
15-epoch curve peaks at epoch six while training loss continues to decrease,
which is consistent with overfitting to the small capture set.

## Decision

Do not run the final held-out test yet. Add training-only sessions rich in
friendly bodies and partial occlusions, retrain with the same architecture, and
use the existing validation split for checkpoint selection. Freeze the revised
checkpoint and thresholds before capturing the final test route. Enemy/friendly
classification remains a later stage; this pilot detects any visible body and
reports recall grouped by the retained team labels.
