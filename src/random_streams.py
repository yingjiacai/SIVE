"""Deterministic, independent random streams for localized probes."""

import torch


_MAX_SEED = 2**63 - 1
_STREAM_OFFSETS = {
    "gradient": 104_729,
    "langevin": 130_363,
    "evaluation": 155_921,
}


def derive_probe_seed(base_seed, checkpoint, stream, trajectory=0):
    """Derive a stable seed without relying on Python's randomized ``hash``."""
    if stream not in _STREAM_OFFSETS:
        raise KeyError(f"Unknown probe random stream: {stream}")
    value = (
        (int(base_seed) + 1) * 1_000_003
        + (int(trajectory) + 1) * 10_000_019
        + (int(checkpoint) + 1) * 100_003
        + _STREAM_OFFSETS[stream]
    )
    return value % _MAX_SEED


def make_torch_generator(device, seed):
    """Create a generator on the same device type as the requested draws."""
    device = torch.device(device)
    generator = torch.Generator(device=str(device))
    generator.manual_seed(int(seed))
    return generator


def make_probe_rng_streams(base_seed, checkpoint, device, trajectory=0):
    """Return mutually independent gradient, Langevin, and evaluation streams.

    The seed recipe deliberately excludes the localization multiplier. Runs at
    different ``c_h`` values therefore consume common random numbers while
    retaining independence between the three stochastic mechanisms.
    """
    return {
        name: make_torch_generator(
            device,
            derive_probe_seed(base_seed, checkpoint, name, trajectory=trajectory),
        )
        for name in _STREAM_OFFSETS
    }


def probe_seed_manifest(base_seed, checkpoint, trajectory=0):
    """Return serializable stream seeds for provenance and debugging."""
    return {
        name: derive_probe_seed(base_seed, checkpoint, name, trajectory=trajectory)
        for name in _STREAM_OFFSETS
    }
