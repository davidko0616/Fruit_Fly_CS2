import copy
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from models.flywire_network import FlyWireNetwork
from training.recording import ActivityRecorder, read_events, atomic_json
from training.train_recorded import load_version
from visualization.serve import Recording


class RecordingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / 'recording'
        adjacency = sp.csr_matrix(np.ones((4, 4), dtype=np.float32))
        self.model = FlyWireNetwork(adjacency, np.array([1, -1, 1, 1]), [0], [3], 2, 3, 3)
        self.ids = [720575940000000001 + i for i in range(4)]
        self.x = torch.tensor([[0.1, 0.2], [0.3, -0.4]])
        self.y = torch.tensor([1, 2])

    def record(self, recorder, model=None):
        device = next((model or self.model).parameters()).device
        return recorder.forward(self.x.to(device), self.y.to(device), [7, 8],
                                split='train', phase='optimization', epoch=1, version=0)

    def check_equivalence(self, device):
        self.model.to(device)
        control = copy.deepcopy(self.model)
        x, y = self.x.to(device), self.y.to(device)
        expected = control(x)
        F.cross_entropy(expected, y).backward()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
        with ActivityRecorder(self.directory, self.model, self.ids, {}) as recorder:
            recorder.save_weights(0)
            actual = self.record(recorder)
            F.cross_entropy(actual, y).backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for observed, reference in zip(self.model.parameters(), control.parameters()):
            torch.testing.assert_close(observed.grad, reference.grad, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        if cuda_rng is not None:
            torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
        reader = Recording(self.directory)
        self.assertEqual(reader.graph['root_ids'][0], str(self.ids[0]))
        frame = reader.frame(0, 1)
        self.assertEqual(frame['sample_ids'], 8)
        np.testing.assert_array_equal(frame['logits'], actual[1].detach().cpu().numpy())
        with torch.no_grad():
            state = torch.zeros(2, 4, device=device)
            state[:, [0]] = control.input_proj(x)
            np.testing.assert_array_equal(frame['states'][0], state[1].cpu().numpy())
            for step in range(3):
                pre = control.connectome_layer(state)
                state = torch.relu(pre)
                np.testing.assert_array_equal(frame['preactivations'][step], pre[1].cpu().numpy())
                np.testing.assert_array_equal(frame['states'][step + 1], state[1].cpu().numpy())

    def test_cpu_outputs_gradients_rng_and_saved_values(self):
        self.check_equivalence('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_outputs_gradients_rng_and_saved_values(self):
        self.check_equivalence('cuda')

    def test_interruption_flushes_and_stale_manifest_recovery(self):
        with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
            with ActivityRecorder(self.directory, self.model, self.ids, {}, max_buffer_samples=3) as recorder:
                recorder.save_weights(0)
                self.record(recorder)
                self.record(recorder)
                raise RuntimeError('simulated interruption')
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'interrupted')
        self.assertEqual(len(read_events(self.directory)), 2)
        manifest['committed_chunks'] = 0
        (self.directory / 'manifest.json').write_text(json.dumps(manifest))
        (self.directory / 'chunks' / '000002.npz.tmp').write_bytes(b'incomplete')
        reader = Recording(self.directory)
        self.assertEqual(len(reader.events), 2)
        self.assertEqual(reader.frame(1, 0)['sample_ids'], 7)
        with self.assertRaises(ValueError):
            reader.frame(-1, 0)
        with self.assertRaises(ValueError):
            reader.frame(0, 2)
        with self.assertRaises(FileExistsError):
            ActivityRecorder(self.directory, self.model, self.ids, {})

    def test_writer_failure_stops_recording_and_keeps_completed_chunks(self):
        recorder = ActivityRecorder(self.directory, self.model, self.ids, {}, max_buffer_samples=2)
        recorder.save_weights(0)
        self.record(recorder)
        with patch('training.recording.atomic_npz', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                with recorder:
                    self.record(recorder)
        self.assertEqual(len(read_events(self.directory)), 1)
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'interrupted')

    def test_weight_versions_reproduce_forward_and_diagnostics(self):
        with ActivityRecorder(self.directory, self.model, self.ids, {}) as recorder:
            recorder.save_weights(0)
            expected = self.record(recorder).detach().clone()
            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter.add_(0.1)
            recorder.save_weights(1, {'from_version': 0, 'loss': 1.5})
            load_version(self.model, self.directory, 0)
            torch.testing.assert_close(self.model(self.x), expected, rtol=0, atol=0)
        with np.load(self.directory / 'weights/0000001.npz') as data:
            self.assertEqual(json.loads(str(data['metadata']))['loss'], 1.5)

    def test_transient_file_lock_is_retried_without_losing_manifest(self):
        path = Path(self.temporary.name) / 'manifest.json'
        atomic_json(path, {'version': 0})
        replace = os.replace
        calls = []

        def transient(source, destination):
            calls.append(1)
            if len(calls) < 3:
                raise PermissionError('File locked')
            return replace(source, destination)

        with patch('training.recording.os.replace', side_effect=transient), patch('training.recording.time.sleep'):
            atomic_json(path, {'version': 1})
        self.assertEqual(len(calls), 3)
        self.assertEqual(json.loads(path.read_text())['version'], 1)

    def test_persistent_file_lock_preserves_previous_manifest(self):
        path = Path(self.temporary.name) / 'manifest.json'
        atomic_json(path, {'version': 0})
        with patch('training.recording.os.replace', side_effect=PermissionError('File locked')), patch('training.recording.time.sleep'):
            with self.assertRaises(PermissionError):
                atomic_json(path, {'version': 1})
        self.assertEqual(json.loads(path.read_text())['version'], 0)

    def test_missing_last_chunk_is_not_presented_as_complete(self):
        with ActivityRecorder(self.directory, self.model, self.ids, {}, max_buffer_samples=2) as recorder:
            recorder.save_weights(0)
            self.record(recorder)
            self.record(recorder)
        (self.directory / 'chunks/000001.npz').rename(self.directory / 'chunks/000001.npz.tmp')
        with self.assertRaisesRegex(ValueError, 'missing committed'):
            Recording(self.directory)


if __name__ == '__main__':
    unittest.main()
