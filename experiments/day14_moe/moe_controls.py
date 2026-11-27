"""Day 14 - implementations by name (ref, mine) and the negative controls: the reference with ONE thing changed.

EXPECTED lists, per control, the checks that MUST fail on the `ragged` routing case (small preset, 2 layers);
every other check must pass. A control whose failures differ from EXPECTED is "not as predicted": either the
harness or the explanation is wrong (Day 11's rule). Derivations: GUIDE Block 2 Stage 4.

Check names produced by check_ep.py:
  routing.L{k}  counts.L{k}  logits  loss.ce  loss.aux.L{k}  loss.z.L{k}  dx  grad.dense  grad.router.L{k}
  grad.expert.L{k}.e{j}  replicated  collectives
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
import torch.distributed as dist

import moe_ref

NAMES = ("top1_route", "permute", "unpermute", "exchange_counts", "AllToAllV", "finish_grads")


def _ns(**override):
    base = {n: getattr(moe_ref, n) for n in NAMES}
    base.update(override)
    return SimpleNamespace(**base)


# 1. wrong_inverse: "un"-permute by gathering with the same order instead of scattering into it.
def unpermute_wrong_inverse(y_sorted, order, n_tokens):
    return y_sorted[order]               # <- the bug: applies the permutation a second time instead of inverting it


# 2. expert_sum_over_ep: treats the EP ranks' experts as if they were DDP replicas and sums them.
def finish_grads_sum_over_ep(model, groups):
    moe_ref.finish_grads(model, groups)
    for p in model.parameters():
        if getattr(p, "expert_parallel", False):
            dist.all_reduce(p.grad, group=groups.ep_group)   # <- the bug: equal shapes, DIFFERENT experts
            p.grad /= groups.ep


# 3. expert_no_dp_scale: forgets that the expert grad is a sum over DP source ranks.
def finish_grads_no_dp_scale(model, groups):
    for p in model.parameters():
        if getattr(p, "expert_parallel", False) and p.grad is not None:
            p.grad.mul_(groups.dp)                           # undo the reference's 1/DP below...
    moe_ref.finish_grads(model, groups)                      # <- net effect: expert grads not divided by DP


# 4. recv_equals_send: skips the count exchange and assumes "I receive what I send" (symmetric traffic).
def exchange_counts_assume_symmetric(send_counts, group):
    return send_counts.clone()                               # <- the bug: no all_to_all of counts


CONTROLS = {
    "wrong_inverse": (_ns(unpermute=unpermute_wrong_inverse), "ragged"),
    # The same bug on data where every rank's tokens are already grouped by expert: x[order] == x, so the
    # wrong inverse is the right one. A check on such data cannot see the bug: EXPECTED is empty.
    "wrong_inverse@presorted": (_ns(unpermute=unpermute_wrong_inverse), "presorted"),
    "expert_sum_over_ep": (_ns(finish_grads=finish_grads_sum_over_ep), "ragged"),
    "expert_no_dp_scale": (_ns(finish_grads=finish_grads_no_dp_scale), "ragged"),
    "recv_equals_send": (_ns(exchange_counts=exchange_counts_assume_symmetric), "ragged"),
}

_E = (0, 1, 3)   # experts that receive tokens in `ragged`; expert 2 gets none, so its gradient is 0 in every variant
EXPECTED = {
    # Layer 0's MoE output lands on the wrong rows -> everything downstream of it: the logits and loss, layer 1's
    # router inputs (so its aux/z terms), and every gradient that depends on those rows. Layer 0's aux and z terms
    # are computed before the bug, routing and counts are forced, and expert 2 sees no token.
    "wrong_inverse": ["logits", "loss.ce", "dx", "loss.aux.L1", "loss.z.L1", "grad.dense",
                      "grad.router.L0", "grad.router.L1"]
                     + [f"grad.expert.L{k}.e{e}" for k in (0, 1) for e in _E],
    "wrong_inverse@presorted": [],
    # every expert grad becomes the mean of four different experts' grads; expert 2 is no longer zero
    "expert_sum_over_ep": [f"grad.expert.L{k}.e{e}" for k in (0, 1) for e in range(4)],
    # a clean x4 (ratio 4.000) on every expert that received tokens
    "expert_no_dp_scale": [f"grad.expert.L{k}.e{e}" for k in (0, 1) for e in _E],
    # caught by the split validator on every rank, before the payload all_to_all
    "recv_equals_send": ["raised:SplitMetadataError"],
}


def todo_left(module) -> list[str]:
    """Names of functions in `module` whose source still contains `raise NotImplementedError`."""
    left = []
    for name in NAMES:
        obj = getattr(module, name)
        parts = [obj.forward, obj.backward] if inspect.isclass(obj) else [obj]
        for part in parts:
            if "raise NotImplementedError" in inspect.getsource(part):
                left.append(name if not inspect.isclass(obj) else f"{name}.{part.__name__}")
    return left


def load_impl(name: str):
    """ref | mine | a control name -> (namespace with the six names, forced pattern or None)."""
    if name == "ref":
        return _ns(), None
    if name == "mine":
        import my_moe
        return SimpleNamespace(**{n: getattr(my_moe, n) for n in NAMES}), None
    if name in CONTROLS:
        return CONTROLS[name]
    raise ValueError(f"unknown impl {name!r}: ref, mine or one of {sorted(CONTROLS)}")


def local_split_controls(send: torch.Tensor) -> dict:
    """Split-metadata controls on 4 LOGICAL ranks in one process [simulated]: no process group, no NCCL.

    send [src, dst] token counts. Each control builds every rank's (in_splits, out_splits) lists with one mistake;
    moe_ref.split_problems must flag it, and must flag nothing for the correct lists.
    """
    ep = send.shape[0]
    rows = send.sum(1).tolist()
    correct_in = [send[r].tolist() for r in range(ep)]
    correct_out = [send[:, r].tolist() for r in range(ep)]
    variants = {
        "reference": (correct_in, correct_out),
        "recv_equals_send": (correct_in, correct_in),                                     # assume symmetric
        "drop_zero_splits": ([[c for c in s if c] for s in correct_in],                   # "skip empty peers"
                             [[c for c in s if c] for s in correct_out]),
        "transposed_matrix": ([send[:, r].tolist() for r in range(ep)],                     # read M by column
                              [send[r].tolist() for r in range(ep)]),
    }
    return {k: moe_ref.split_problems(i, o, rows) for k, (i, o) in variants.items()}
