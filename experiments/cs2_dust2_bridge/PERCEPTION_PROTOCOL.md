# Dust II visible-perception protocol

This stage builds the screen-visible target signal required by the read-only
bridge. It must not use `allplayers`, game memory, server plugins, demos, radar-only
enemy positions, or any other source that can reveal a player behind geometry.

## Capture

- Keep CS2 at 2560 x 1440 and use a full-screen crop of `(0, 0, 2560, 1440)`.
- Use DXcam on Windows when Pillow cannot capture the Direct3D surface. DXcam uses
  the visible display output and does not require an NVIDIA GPU.
- Record separate sessions for train, validation, and test. Never distribute
  adjacent frames from one capture across different splits.
- Include full and partial bodies, near and far players, varied lighting and map
  areas, weapon/hand occlusion, and verified frames with no visible player.
- Keep Codex, Steam, console, and desktop overlays out of accepted frames.

```powershell
.\.venv\Scripts\python.exe tools/capture_cs2_screen.py --output artifacts/cs2_bridge/dust2_visible_players_train_01 --frames 300 --interval 0.2 --region 0,0,2560,1440 --backend dxcam --delay 20 --sound-cues --spoken-prompt "Keep visible players on screen at several distances."
```

On Windows, the optional spoken prompt plays before the delay. Three countdown
tones lead into a distinct start tone; a rising completion sequence and spoken
completion message indicate that it is safe to return to Codex. A descending
sequence and failure message report capture errors. No overlay is drawn into
captured frames.

The first local 300-frame capture validates the DXGI path and includes useful
full-body and partial-view examples. It also contains display-transition and
Codex-overlay frames; those frames must remain unlabeled and excluded from model
data. The first review retained 19 valid enemy boxes: 16 full-body and three
partial-body examples. This is a pipeline pilot, not enough data to train or
evaluate a detector. A subsequent clean Long A session contributed nine evenly
spaced, audited hard-negative frames containing walls, a car, doors, shadows,
and no visible player body.

## Visible-player labels

Run the loopback-only labeler and draw a tight box around every player whose body
is visible in the game view. Record enemy, friendly, or unknown and full or
partial visibility. Use **Mark none** only for a verified game frame with no
visible body. Skip transition, white, desktop, and overlay frames.

```powershell
.\.venv\Scripts\python.exe tools/label_cs2_frames.py --capture artifacts/cs2_bridge/dust2_visible_players_train_01 --output artifacts/cs2_bridge/dust2_visible_players_train_01_labels.json --split train --start-frame 0 --end-frame 240 --stride 10
```

The label schema validates positive-area boxes inside the source frame. The
server binds to loopback, serves only files named by the capture manifest, and
writes labels atomically after every saved frame. For live matches, review the
sampled contact sheet before labeling and omit console, death, round-end,
freeze-time, respawn, and buy-menu frames with repeated
`--exclude-frame FRAME_ID` arguments. Those states are handled by synchronized
GSI activity and round state and must not be recorded as ordinary detector
negatives.

Audit the labels before using them:

```powershell
.\.venv\Scripts\python.exe tools/audit_cs2_labels.py --capture artifacts/cs2_bridge/dust2_visible_players_train_01 --labels artifacts/cs2_bridge/dust2_visible_players_train_01_labels.json
```

## Detector dataset export

Export only audited sessions. Each capture keeps its declared split, so adjacent
frames from one recording cannot leak between training and validation. The
exporter writes YOLO detection labels, creates empty label files for verified
negative frames, and retains source boxes, team, and visibility in JSONL
metadata. Hard links avoid duplicating the full-resolution PNG data when the
source and output are on the same filesystem.

```powershell
.\.venv\Scripts\python.exe tools/export_cs2_detector_dataset.py `
  --session artifacts/cs2_bridge/dust2_visible_players_train_01 artifacts/cs2_bridge/dust2_visible_players_train_01_labels.json `
  --session artifacts/cs2_bridge/dust2_visible_players_train_neg_02 artifacts/cs2_bridge/dust2_visible_players_train_neg_02_labels.json `
  --session artifacts/cs2_bridge/dust2_visible_players_train_02 artifacts/cs2_bridge/dust2_visible_players_train_02_labels.json `
  --session artifacts/cs2_bridge/dust2_visible_players_train_03 artifacts/cs2_bridge/dust2_visible_players_train_03_labels.json `
  --session artifacts/cs2_bridge/dust2_visible_players_train_04 artifacts/cs2_bridge/dust2_visible_players_train_04_labels.json `
  --session artifacts/cs2_bridge/dust2_visible_players_validation_01 artifacts/cs2_bridge/dust2_visible_players_validation_01_labels.json `
  --output artifacts/cs2_bridge/dust2_detector_player_dataset_v3 `
  --class-mode player --transfer hardlink
```

The corrected pilot export contains 95 training frames with 85 boxes and 36
verified negatives, plus 20 independent validation frames with 11 boxes and 11
verified negatives. The exported dataset remains local under `artifacts/`.
Its size supports an end-to-end detector pilot and error analysis, not a final
accuracy claim. Freeze the detector configuration before collecting or opening
the final held-out test route.

## First CPU detector pilot

The first pilot uses TorchVision 0.22.1's SSDlite320 MobileNetV3 detector with
PyTorch 2.7.1+cpu. Its background/person classifier starts from the matching COCO
weights; the MobileNet backbone stays frozen. Training uses seed 42, batch size
two, AdamW at `1e-4`, and selects the best of 15 epochs by fixed-threshold
validation F1. `tools/train_cs2_detector.py` refuses to overwrite a run and saves
configuration, environment, history, last, and best checkpoints.

```powershell
.\.venv\Scripts\python.exe tools/train_cs2_detector.py `
  --dataset artifacts/cs2_bridge/dust2_detector_player_dataset_v3 `
  --output artifacts/cs2_bridge/dust2_player_detector_pilot_v5_corrected_validation `
  --epochs 15 --batch-size 2 --learning-rate 0.0001 `
  --weight-decay 0.0001 --seed 42 --cpu-threads 6 `
  --image-size 320 --score-threshold 0.25 --iou-threshold 0.5 --freeze-backbone
```

After adding a teammate-heavy training session and correcting one loose
validation box, epoch nine was selected. Validation-only calibration froze score threshold 0.25,
NMS threshold 0.30, and IoU threshold 0.50 before any test capture. On the 20
validation frames, the selected detector has 53.8% precision, 63.6% recall, F1
0.583, zero detections on 11 negative frames, and about 35 ms mean CPU latency.
Grouped recall is 83.3% for enemies, 40.0% for teammates, 85.7% for full bodies,
and 25.0% for partial bodies. A matched 640-pixel run and full-frame-plus-tiles
inference both performed worse. These are pilot diagnostics, not held-out test
evidence; see [the full report](DETECTOR_PILOT.md).

Run fixed-threshold validation or test evaluation with:

```powershell
.\.venv\Scripts\python.exe tools/evaluate_cs2_detector.py `
  --dataset artifacts/cs2_bridge/dust2_detector_player_dataset_v3 `
  --checkpoint artifacts/cs2_bridge/dust2_player_detector_pilot_v5_corrected_validation/best.pt `
  --split validation --thresholds 0.25 --nms-threshold 0.30 `
  --iou-threshold 0.50
```

## Acceptance before controller use

Freeze the detector configuration before evaluating it on the held-out sessions.
Report person-level precision and recall, enemy/friendly confusion, full versus
partial recall, false positives per negative frame, and inference latency on CPU.
Then calibrate visible-target bearing and range from held-out visible examples.
The target field must be null whenever no body is visibly detected, even if the
radar retains a spotted-player icon. Last-seen memory may continue only through
the bridge's existing round-scoped memory path.

Local clearance will be evaluated separately from a static Dust II radar mask
and the already calibrated own-player pose. It cannot use opponent markers and
must report ray-cast error at manually checked locations before supplying the
four controller clearance values.
