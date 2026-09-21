"""The model reading out statistics about its own weights."""


import torch


def _fmt(x):
    return f"{x:.4g}"


@torch.no_grad()
def weight_stats(model, step=None, loss=None, lr=None, gnorm=None):
    cfg = model.cfg
    stats = {
        "step": step,
        "params": model.n_params(),
        "params_non_embedding": model.n_params(non_embedding=True),
        "d_model": cfg.d_model,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "block": cfg.block,
        "loss": loss,
        "lr": lr,
        "grad_norm": gnorm,
        "groups": [],
    }

    def group(name, tensors):
        tensors = [t for t in tensors if t is not None]
        if not tensors:
            return
        flat = torch.cat([t.detach().float().reshape(-1) for t in tensors])
        stats["groups"].append({
            "name": name,
            "n": flat.numel(),
            "rms": float(flat.pow(2).mean().sqrt()),
            "absmax": float(flat.abs().max()),
            "mean": float(flat.mean()),
            "near_zero": float((flat.abs() < 1e-3).float().mean()),
        })

    group("emb", [model.tok_emb.weight])
    for i, b in enumerate(model.blocks):
        # collect by module rather than by attribute name: a sparse block's
        # mlp is a router plus a bank of experts, not w1/w2/w3
        group(f"l{i}.attn", list(b.attn.parameters()))
        group(f"l{i}.mlp", list(b.mlp.parameters()))
        group(f"l{i}.norm", [b.ln1.weight, b.ln2.weight])
    group("ln_f", [model.ln_f.weight])
    return stats


def render_report(stats, full=False):
    """Canonical <self> block. Compact by default so it is cheap to mix in."""
    head = (f"model mini-AGI params {stats['params'] / 1e6:.1f}M "
            f"d{stats['d_model']} L{stats['n_layer']} H{stats['n_head']} "
            f"ctx{stats['block']}")
    lines = ["<self>", head]
    if stats.get("step") is not None:
        tail = f"step {stats['step']}"
        if stats.get("loss") is not None:
            tail += f" loss {stats['loss']:.4f}"
        if stats.get("lr") is not None:
            tail += f" lr {stats['lr']:.2e}"
        if stats.get("grad_norm") is not None:
            tail += f" gnorm {stats['grad_norm']:.3f}"
        lines.append(tail)
    groups = stats["groups"]
    if not full:
        # first, middle and last layer is enough to show the shape of the model
        keep = {"emb", "ln_f"}
        L = stats["n_layer"]
        for i in (0, L // 2, L - 1):
            keep |= {f"l{i}.attn", f"l{i}.mlp"}
        groups = [g for g in groups if g["name"] in keep]
    for g in groups:
        lines.append(f"{g['name']} n {g['n']} rms {_fmt(g['rms'])} "
                     f"absmax {_fmt(g['absmax'])} mean {_fmt(g['mean'])} "
                     f"zero {g['near_zero']:.3f}")
    lines.append("</self>")
    return "\n".join(lines) + "\n"


