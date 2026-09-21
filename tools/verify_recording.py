"""Audit recorded data and reproduce selected forwards in a separate process."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.flywire_network import FlyWireNetwork
from training.recording import read_events
from training.train_recorded import load_version
from visualization.serve import Recording


def verify(directory):
    directory = Path(directory)
    reader = Recording(directory)
    events = read_events(directory)
    if not events:
        raise ValueError('No committed events')
    with np.load(directory / 'dataset.npz') as file:
        dataset = dict(file)
    samples = 0
    for chunk_name in dict.fromkeys(event['chunk'] for event in events):
        with np.load(directory / 'chunks' / chunk_name, allow_pickle=False) as chunk:
            chunk_events = json.loads(str(chunk['events']))
            count = sum(e['count'] for e in chunk_events)
            for key in chunk.files:
                if key != 'events':
                    if len(chunk[key]) != count or not np.isfinite(chunk[key]).all():
                        raise ValueError(f'Invalid array {chunk_name}/{key}')
            offset = 0
            for event in chunk_events:
                if event['offset'] != offset:
                    raise ValueError('Invalid event offsets')
                section = slice(offset, offset + event['count'])
                ids = chunk['sample_ids'][section]
                if not np.isin(ids, dataset[event['split']]).all():
                    raise ValueError('Sample assigned to wrong split')
                np.testing.assert_array_equal(chunk['inputs'][section], dataset['x'][ids])
                np.testing.assert_array_equal(chunk['labels'][section], dataset['y'][ids])
                np.testing.assert_array_equal(chunk['predictions'][section], chunk['logits'][section].argmax(1))
                if not (directory / 'weights' / f'{event["version"]:07d}.npz').is_file():
                    raise ValueError('Missing parameter version')
                offset += event['count']
            samples += count
    manifest = reader.manifest
    if manifest['status'] == 'complete' and (samples != manifest['samples'] or len(events) != manifest['events']):
        raise ValueError('Final manifest disagrees with committed data')
    for epoch in range(1, manifest['config']['epochs'] + 1):
        selected = [e for e in events if e['epoch'] == epoch and e['phase'] == 'optimization']
        if not selected:
            continue
        ids = np.concatenate([reader.chunk(e['chunk'])['sample_ids'][e['offset']:e['offset']+e['count']]
                              for e in selected])
        if manifest['status'] == 'complete':
            np.testing.assert_array_equal(np.sort(ids), np.sort(dataset['train']))
        elif len(np.unique(ids)) != len(ids):
            raise ValueError('Duplicate optimization samples in interrupted epoch')
    with np.load(directory / 'graph.npz') as graph:
        model = FlyWireNetwork(sp.load_npz(directory / 'adjacency.npz'), graph['signs'],
                               graph['inputs'], graph['outputs'], 2, 3, manifest['num_steps'],
                               'normalized_synapse_count')
    torch.set_num_threads(4)
    # Preserve original batch size: sparse kernels can round differently across sizes.
    checked = sorted({0, len(events) // 2, len(events) - 1})
    for index in checked:
        event = events[index]
        chunk = reader.chunk(event['chunk'])
        section = slice(event['offset'], event['offset'] + event['count'])
        load_version(model, directory, event['version'])
        with torch.no_grad():
            x = torch.from_numpy(chunk['inputs'][section])
            observed = model(x).numpy()
        np.testing.assert_allclose(observed, chunk['logits'][section], rtol=1e-4, atol=1e-5)
    result = {'status': manifest['status'], 'events': len(events), 'sample_forwards': samples,
              'reproduced_event_ids': checked, 'audit_passed': True}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    verify(parser.parse_args().run)
