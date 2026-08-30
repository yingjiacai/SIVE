"""Configuration helpers shared by the SIVE experiment scripts."""

from copy import deepcopy


def resolve_sgld_config(config):
    """Return a validated copy with explicit ``t`` and SGLD step size ``lr``.

    New experiment files should specify ``t`` directly.  The ``n * beta``
    fallback is retained only so that archived configurations remain readable.
    """
    resolved = deepcopy(config)

    if "t" not in resolved:
        if "n" not in resolved or "beta" not in resolved:
            raise KeyError("SGLD configuration must define an explicit positive 't'.")
        resolved["t"] = resolved["n"] * resolved["beta"]
        resolved["legacy_t_source"] = "n * beta"

    resolved["t"] = float(resolved["t"])
    if resolved["t"] <= 0:
        raise ValueError("'t' must be positive.")

    if "base_lr" in resolved:
        resolved["lr"] = float(resolved["base_lr"]) / resolved["t"]
    elif "lr" not in resolved:
        raise KeyError("SGLD configuration must define 'base_lr' or 'lr'.")

    resolved["lr"] = float(resolved["lr"])
    if resolved["lr"] <= 0:
        raise ValueError("The resolved SGLD step size 'lr' must be positive.")
    if int(resolved.get("M", 0)) < 2:
        raise ValueError("'M' must be at least 2 for a sample variance.")
    if int(resolved.get("N", 0)) < 2:
        raise ValueError("'N' must be at least 2 for replicated-noise debiasing.")
    if resolved.get("model") == "Mlp":
        if "c_h" not in resolved:
            if "h" not in resolved:
                raise KeyError("MLP configurations must define a positive 'c_h'.")
            resolved["c_h"] = resolved["h"]
            resolved["legacy_c_h_source"] = "h"
        resolved["c_h"] = float(resolved["c_h"])
        if resolved["c_h"] <= 0:
            raise ValueError("'c_h' must be positive.")
    else:
        if float(resolved.get("h", 0.0)) <= 0:
            raise ValueError("'h' must be positive.")
        resolved["h"] = float(resolved["h"])

    resolved["M"] = int(resolved["M"])
    resolved["N"] = int(resolved["N"])
    resolved["schema_version"] = int(resolved.get("schema_version", 2))
    return resolved
