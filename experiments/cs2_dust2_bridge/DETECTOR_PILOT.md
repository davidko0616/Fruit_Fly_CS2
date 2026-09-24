# Dust II visible-player detector pilot

## Scope

This is an intermediate CPU feasibility result for screen-visible player
detection. It does not establish final accuracy and has not been evaluated on
the held-out test route. The detector receives captured RGB frames only; radar
contacts and game-state opponent coordinates are not inputs.

The accepted local dataset contains five training sessions and one independent
validation session:

- train: 95 frames, 59 positive frames, 36 verified negatives, 85 boxes;
- validation: 20 frames, nine positive frames, 11 verified negatives, 11 boxes.

The latest training-only live-match session added 20 active-play frames: 15
positive frames, five verified negatives, and 34 boxes. Thirty-three boxes are
friendly players and 12 are partially visible. Console, round-end, and buy-menu
frames were explicitly excluded rather than treated as negatives. A validation
QA pass tightened one erroneous close-player box before the final run.

Capture sessions remain intact within one split. Full-resolution images and
checkpoints remain local under `artifacts/cs2_bridge/`.

## Fixed 320-pixel run

- model: TorchVision SSDlite320 MobileNetV3 Large;
- initialization: official COCO background/person weights;
- runtime: PyTorch 2.7.1+cpu and TorchVision 0.22.1+cpu;
- input: 320 x 320;
- seed: 42;
- optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`;
- batch size: two, 15 epochs, frozen MobileNet backbone;
- checkpoint selection: validation F1 at score 0.25 and IoU 0.50;
- selected checkpoint: epoch nine.

The official PyTorch compatibility table pairs PyTorch 2.7.1 with TorchVision
0.22.1. TorchVision describes SSDlite320 as a compact detector intended for
efficient inference:

- <https://pytorch.org/get-started/previous-versions/>
- <https://pytorch.org/blog/torchvision-ssdlite-implementation/>

## Validation result

Validation-only calibration selects score threshold 0.25, NMS threshold 0.30,
and match IoU 0.50. At those settings:

| Metric | Result |
|---|---:|
| Precision | 53.8% |
| Recall | 63.6% |
| F1 | 0.583 |
| Enemy recall | 83.3% (5/6) |
| Friendly recall | 40.0% (2/5) |
| Full-body recall | 85.7% (6/7) |
| Partial-body recall | 25.0% (1/4) |
| False positives on negative frames | 0 across 11 frames |
| Mean CPU latency | about 35 ms/frame |

The earlier pre-targeted-data pilot reached 40.0% precision, 54.5% recall, and
F1 0.462 on the same validation session. The added teammate-heavy session and
validation correction therefore improved the pilot, but partial-player recall
remains poor. The four remaining misses comprise three partial bodies and one
small full body. Two targets are only 55 x 80 and 44 x 33 pixels in the original
2560 x 1440 frame, so they become only a few pixels after 320 x 320 resizing.

## Rejected resolution experiments

Two validation-only experiments were retained as negative results:

- A 640 x 640 run used the same training data, seed, optimizer, and 15-epoch
  schedule. Its best calibrated F1 was 0.400 and mean CPU latency was about
  57 ms/frame. It produced more false positives and lower recall than the fixed
  320-pixel model.
- Full-frame plus four 320-pixel quadrant inferences preserved more small-target
  pixels, but its best validation F1 was 0.381 at about 96 ms/frame and it added
  false detections on negative frames.

Neither experiment replaces the selected 320-pixel checkpoint.

## Team-aware targeting gate

The same audited sessions were exported with `enemy`, `friendly`, and `unknown`
classes. The training boxes are balanced: 43 enemy and 42 friendly. A first
frozen-backbone run underperformed, so the final pilot fine-tuned the complete
320-pixel network for 15 epochs with the same seed and optimizer. Validation F1
at score threshold 0.15 selected epoch five; NMS was then fixed at 0.30.

| Team-aware metric | Result |
|---|---:|
| Overall precision | 63.6% |
| Overall recall | 63.6% |
| Overall F1 | 0.636 |
| Enemy precision | 71.4% (5/7) |
| Enemy recall | 83.3% (5/6) |
| Friendly precision | 50.0% (2/4) |
| Friendly recall | 40.0% (2/5) |
| Correct team among matched players | 85.7% (6/7) |
| False positives on negative frames | 0 across 11 frames |
| Mean CPU latency | about 36 ms/frame |

One matched enemy was classified as friendly. No matched friendly was classified
as enemy. Four unmatched or duplicate predictions remain on positive frames, so
these figures support read-only integration only; they do not authorize firing.

## Decision

Keep the team-aware 320-pixel epoch-five checkpoint as the current read-only
integration baseline, with score threshold 0.15 and NMS 0.30. Do not run the
final held-out detector test yet. First synchronize its detections with
fixed-radar pose and GSI round state, verify that targets become null on misses
and occlusion, and replay the combined observation stream without game input.
A later detector revision should target partial and tiny players with an
architecture designed for multi-scale detection rather than stretching
SSDlite320 or merging uncalibrated tiled predictions.
