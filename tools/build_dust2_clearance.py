"""Build a Dust II walkability mask and register it to radar player poses."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, optimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _read_glb_mesh(path):
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != b'glTF':
        raise ValueError('Input is not a binary glTF file')
    _, version, total_length = struct.unpack_from('<4sII', data)
    if version != 2 or total_length != len(data):
        raise ValueError('Only complete glTF 2.0 binary files are supported')
    document = binary = None
    offset = 12
    while offset < len(data):
        length, chunk_type = struct.unpack_from('<I4s', data, offset)
        offset += 8
        chunk = data[offset:offset + length]
        offset += length
        if chunk_type == b'JSON':
            document = json.loads(chunk)
        elif chunk_type == b'BIN\0':
            binary = chunk
    if document is None or binary is None:
        raise ValueError('GLB must contain JSON and binary chunks')
    primitive = document['meshes'][0]['primitives'][0]

    def accessor(index, dtype, width):
        definition = document['accessors'][index]
        view = document['bufferViews'][definition['bufferView']]
        start = int(view.get('byteOffset', 0)) + int(definition.get('byteOffset', 0))
        values = np.frombuffer(binary, dtype=dtype,
                               count=int(definition['count']) * width,
                               offset=start)
        return values.reshape(-1, width)

    positions = accessor(primitive['attributes']['POSITION'], '<f4', 3)
    indices = accessor(primitive['indices'], '<u4', 1).reshape(-1, 3)
    return positions, indices


def rasterize_nav_mesh(glb_path, overview_pos_x=-2476.0,
                       overview_pos_y=3239.0, overview_scale=4.4,
                       mask_size=1024):
    positions, triangles = _read_glb_mesh(glb_path)
    texture = np.column_stack((
        (positions[:, 0] - overview_pos_x) / overview_scale,
        (overview_pos_y - positions[:, 1]) / overview_scale,
    ))
    image = Image.new('L', (mask_size, mask_size), 0)
    draw = ImageDraw.Draw(image)
    for triangle in triangles:
        draw.polygon([tuple(texture[index]) for index in triangle], fill=255)
    return image, positions, triangles


def _load_poses(path):
    rows = [json.loads(line) for line in Path(path).read_text(
        encoding='utf-8').splitlines() if line.strip()]
    poses = np.asarray([[row['radar_pose']['x'], row['radar_pose']['y']]
                        for row in rows], dtype=float)
    if len(poses) < 100:
        raise ValueError('At least 100 radar poses are required for registration')
    return poses


def fit_registration(mask, poses):
    walkable = np.asarray(mask) > 0
    signed_distance = (ndimage.distance_transform_edt(walkable) -
                       ndimage.distance_transform_edt(~walkable))
    training = poses[::2]

    def objective(parameters):
        scale_x, scale_y, offset_x, offset_y = parameters
        texture_x = (training[:, 0] - offset_x) / scale_x
        texture_y = (training[:, 1] - offset_y) / scale_y
        distances = ndimage.map_coordinates(
            signed_distance, [texture_y, texture_x], order=1,
            mode='constant', cval=-50)
        return -float(np.mean(np.clip(distances, -20, 10)))

    result = optimize.differential_evolution(
        objective,
        bounds=((0.27, 0.36), (0.20, 0.28), (160, 210), (115, 160)),
        seed=4, popsize=18, maxiter=150, polish=True, tol=1e-9)
    if not result.success:
        raise RuntimeError(f'Registration failed: {result.message}')

    def evaluate(selected):
        scale_x, scale_y, offset_x, offset_y = result.x
        texture_x = (selected[:, 0] - offset_x) / scale_x
        texture_y = (selected[:, 1] - offset_y) / scale_y
        distances = ndimage.map_coordinates(
            signed_distance, [texture_y, texture_x], order=1,
            mode='constant', cval=-50)
        return {
            'poses': int(len(selected)),
            'inside_rate': float(np.mean(distances > 0)),
            'signed_distance_pixels': {
                name: float(value) for name, value in zip(
                    ('min', 'p05', 'median', 'p95', 'max'),
                    np.quantile(distances, (0, .05, .5, .95, 1)))
            },
        }

    return result.x, evaluate(training), evaluate(poses[1::2]), evaluate(poses)


def build(glb_path, poses_path, output_calibration, output_mask,
          overview_pos_x=-2476.0, overview_pos_y=3239.0,
          overview_scale=4.4, max_distance_world=600.0,
          max_snap_world=30.0):
    output_calibration, output_mask = (Path(output_calibration), Path(output_mask))
    if output_calibration.exists() or output_mask.exists():
        raise FileExistsError('Refusing to overwrite calibration or mask output')
    mask, positions, triangles = rasterize_nav_mesh(
        glb_path, overview_pos_x, overview_pos_y, overview_scale)
    poses = _load_poses(poses_path)
    parameters, training, validation, overall = fit_registration(mask, poses)
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_mask)
    try:
        relative_mask = output_mask.relative_to(output_calibration.parent)
    except ValueError:
        relative_mask = output_mask.resolve()
    scale_x, scale_y, offset_x, offset_y = parameters
    result = {
        'schema_version': 1,
        'map_name': 'de_dust2',
        'screen_scale_x': float(scale_x),
        'screen_scale_y': float(scale_y),
        'screen_offset_x': float(offset_x),
        'screen_offset_y': float(offset_y),
        'overview_units_per_pixel': float(overview_scale),
        'max_distance_world': float(max_distance_world),
        'max_snap_world': float(max_snap_world),
        'mask_file': str(relative_mask).replace('\\', '/'),
    }
    output_calibration.parent.mkdir(parents=True, exist_ok=True)
    output_calibration.write_text(json.dumps(result, indent=2) + '\n',
                                  encoding='utf-8')
    return {
        'status': 'complete', 'nav_vertices': int(len(positions)),
        'nav_triangles': int(len(triangles)),
        'walkable_pixels': int(np.count_nonzero(mask)),
        'training': training, 'held_out_odd_frames': validation,
        'overall': overall,
        'nav_glb_sha256': hashlib.sha256(Path(glb_path).read_bytes()).hexdigest(),
        'poses_sha256': hashlib.sha256(Path(poses_path).read_bytes()).hexdigest(),
        'calibration': str(output_calibration), 'mask': str(output_mask),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nav-glb', type=Path, required=True)
    parser.add_argument('--poses', type=Path, required=True)
    parser.add_argument('--output-calibration', type=Path, required=True)
    parser.add_argument('--output-mask', type=Path, required=True)
    parser.add_argument('--overview-pos-x', type=float, default=-2476)
    parser.add_argument('--overview-pos-y', type=float, default=3239)
    parser.add_argument('--overview-scale', type=float, default=4.4)
    parser.add_argument('--max-distance-world', type=float, default=600)
    parser.add_argument('--max-snap-world', type=float, default=30)
    args = parser.parse_args()
    print(json.dumps(build(
        args.nav_glb, args.poses, args.output_calibration, args.output_mask,
        args.overview_pos_x, args.overview_pos_y, args.overview_scale,
        args.max_distance_world, args.max_snap_world), indent=2))


if __name__ == '__main__':
    main()
