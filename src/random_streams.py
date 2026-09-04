"""Deterministic, independent random streams for localized probes."""

import torch


_STREAM_IDS = {
    "gradient": 0,
    "langevin": 1,
    "evaluation": 2,
}


def derive_probe_seed(base_seed, checkpoint, stream, trajectory=0):
    """Encode trajectory/checkpoint/probe/stream IDs in one readable seed."""
    if stream not in _STREAM_IDS:
        raise KeyError(f"Unknown probe random stream: {stream}")
    return (
        int(trajectory) * 1_000_000
        + int(checkpoint) * 1_000
        + int(base_seed) * 10
        + _STREAM_IDS[stream]
    )


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
        for name in _STREAM_IDS
    }


def probe_seed_manifest(base_seed, checkpoint, trajectory=0):
    """Return serializable stream seeds for provenance and debugging."""
    return {
        name: derive_probe_seed(base_seed, checkpoint, name, trajectory=trajectory)
        for name in _STREAM_IDS
    }
