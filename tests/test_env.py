import torch
import torchaudio

from lab import device, seed_all


def test_device_available():
    d = device()
    x = torch.randn(4, 4, device=d)
    assert (x @ x).shape == (4, 4)


def test_seed_reproducible():
    seed_all(1); a = torch.randn(3)
    seed_all(1); b = torch.randn(3)
    assert torch.equal(a, b)


def test_mel_pipeline_runs():
    wav = torch.randn(1, 16000)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)(wav)
    assert mel.shape[1] == 64
