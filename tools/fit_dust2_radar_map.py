"""Fit pan-corrected Dust II radar coordinates to the NAV overview mask."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.radar_map import detect_site_anchor_candidates


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding='utf-8').splitlines() if line.strip()]


def _pose(row):
    return row.get('raw_radar_pose') or row.get('radar_pose')


def _site_pair(candidates):
    for site_b in candidates:
        for site_a in candidates:
            dx = site_a[0] - site_b[0]
            dy = site_a[1] - site_b[1]
            if 175 < dx < 230 and abs(dy) < 30:
                return site_b, site_a
    return None


def _distance_stats(distances):
    distances = np.asarray(distances, dtype=float)
    return {
        'pairs': int(len(distances)),
        'inside_nav_fraction': float(np.mean(distances < .1)),
        'within_30_world_units_fraction': float(np.mean(distances <= 30)),
        'within_90_world_units_fraction': float(np.mean(distances <= 90)),
        'p95_snap_world': float(np.quantile(distances, .95)),
        'maximum_snap_world': float(distances.max()),
    }


def fit(capture, localization, mask_path, output, sample_stride=4,
        overview_units_per_pixel=4.4, x_scale_prior=.337,
        x_scale_prior_weight=20.0, seed=29):
    capture, output = Path(capture), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    rows = [row for row in _rows(localization) if _pose(row) is not None]
    samples = []
    for row in rows[::sample_stride]:
        pose = _pose(row)
        image_path = capture / row['file']
        with Image.open(image_path) as image:
            pair = _site_pair(detect_site_anchor_candidates(image.convert('RGB')))
        if pair is None:
            continue
        samples.append({
            'frame_id': int(row['frame_id']),
            'player': (float(pose['x']), float(pose['y'])),
            'site_b': pair[0], 'site_a': pair[1],
        })
    if len(samples) < 40:
        raise ValueError('At least 40 pose/site-anchor pairs are required')

    mask = np.asarray(Image.open(mask_path).convert('L')) > 0
    signed = (ndimage.distance_transform_edt(mask) -
              ndimage.distance_transform_edt(~mask))
    outside = ndimage.distance_transform_edt(~mask)

    def coordinates(selected, parameters):
        scale_x, scale_y, site_b_x, site_b_y = parameters
        return np.asarray([(
            site_b_x + (row['player'][0] - row['site_b'][0]) / scale_x,
            site_b_y + (row['player'][1] - row['site_b'][1]) / scale_y,
        ) for row in selected])

    training, held_out = samples[::2], samples[1::2]

    def objective(parameters):
        points = coordinates(training, parameters)
        distances = ndimage.map_coordinates(
            signed, [points[:, 1], points[:, 0]], order=1,
            mode='constant', cval=-50)
        data_loss = -float(np.mean(np.clip(distances, -30, 15)))
        prior_loss = x_scale_prior_weight * (
            (parameters[0] - x_scale_prior) / .03) ** 2
        return data_loss + prior_loss

    result = optimize.differential_evolution(
        objective,
        bounds=((.31, .36), (.22, .38), (150, 270), (80, 240)),
        seed=seed, popsize=20, maxiter=200, polish=True, tol=1e-8)
    if not result.success:
        raise RuntimeError(f'Radar-map fit failed: {result.message}')

    def evaluate(selected):
        points = coordinates(selected, result.x)
        valid = ((points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) &
                 (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0]))
        distances = np.full(len(points), np.inf)
        distances[valid] = ndimage.map_coordinates(
            outside, [points[valid, 1], points[valid, 0]], order=1)
        return _distance_stats(distances * overview_units_per_pixel)

    site_dx = np.asarray([
        row['site_a'][0] - row['site_b'][0] for row in samples])
    site_dy = np.asarray([
        row['site_a'][1] - row['site_b'][1] for row in samples])
    scale_x, scale_y, site_b_x, site_b_y = result.x
    document = {
        'schema_version': 1,
        'map_name': 'de_dust2',
        'calibration': {
            'map_name': 'de_dust2',
            'screen_scale_x': float(scale_x),
            'screen_scale_y': float(scale_y),
            'site_b_overview_x': float(site_b_x),
            'site_b_overview_y': float(site_b_y),
            'site_separation_screen_x': float(np.median(site_dx)),
            'site_separation_screen_y': float(np.median(site_dy)),
            'overview_grid_step': 8.0,
            'overview_grid_offset': 4.0,
            'anchor_match_radius': 35.0,
            'anchor_max_age_ms': 2000.0,
        },
        'fit': {
            'source_capture': str(capture),
            'source_localization': str(localization),
            'sample_stride': int(sample_stride),
            'sampled_anchor_pose_pairs': len(samples),
            'fit': evaluate(training),
            'held_out': evaluate(held_out),
            'optimizer_objective': float(result.fun),
            'seed': int(seed),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--localization', type=Path, required=True)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-stride', type=int, default=4)
    parser.add_argument('--overview-units-per-pixel', type=float, default=4.4)
    parser.add_argument('--x-scale-prior', type=float, default=.337)
    parser.add_argument('--x-scale-prior-weight', type=float, default=20)
    parser.add_argument('--seed', type=int, default=29)
    args = parser.parse_args()
    if args.sample_stride < 1 or args.overview_units_per_pixel <= 0:
        parser.error('Stride and overview scale must be positive')
    print(json.dumps(fit(
        args.capture, args.localization, args.mask, args.output,
        args.sample_stride, args.overview_units_per_pixel,
        args.x_scale_prior, args.x_scale_prior_weight, args.seed), indent=2))


if __name__ == '__main__':
    main()
