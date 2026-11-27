"""Day 14 - Day 01's small transformer with each block's MLP replaced by a top-1 MoE layer.

    tokens [B_r, S] -> tok_emb + pos_emb = h0 -> L x [ h += attn(ln1(h));  h += moe(ln2(h)) ] -> norm -> head

B_r = B / 4 sequences on each of the four source ranks (dense DP = 4); the one-GPU baseline holds all B.

One MoE layer, on one rank, for EP > 1 (GUIDE section 2 draws it):

    x [T_r, d] --router--> logits [T_r, E] --top1_route--> idx, w            (replicated router, local tokens)
      --permute--> order, counts [E] --x[order]--> x_sorted                  (grouped by expert = by EP rank)
      --exchange_counts--> recv [EP, E_loc]  --.tolist()-->  splits          (device -> host sync)
      --AllToAllV(out=recv rows, in=send rows)--> x_recv                     (dispatch, uneven)
      --[sort_chunks]--> experts (SequentialMLP loop) --[unsort]--> y_recv
      --AllToAllV(out=send rows, in=recv rows)--> y_sorted                   (combine = dispatch reversed)
      --unpermute--> y [T_r, d] * w                                          (scatter back, weight at the source)

The combine weight w never leaves its source rank: it multiplies the expert output after the combine. Megatron
instead sends the probs along in a second all_to_all and applies them inside the expert (Block 3 Stop 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import moe_ref
from moe_common import MoEConfig, forced_experts, init_value


@dataclass
class RouteOpts:
    pattern: str = "learned"         # see moe_common.PATTERNS
    n_tokens_global: int = 0          # T over all source ranks (forced patterns are defined on the global index)
    aux_scope: str = "global"         # global: f from all ranks' counts (matches one GPU); local: this rank only; off
    validate: bool = True             # check every rank's split lists before the payload all_to_all (GUIDE 2.4)
    capacity: int | None = None       # per-expert capacity C (EP = 1 only; None = dropless)
    drop_policy: str = "position"
    forced_idx: torch.Tensor | None = None   # explicit expert per GLOBAL token id (overrides `pattern`; tests)


class Expert(nn.Module):
    """One FFN expert: w2 . gelu(w1 . x + b1) + b2, PyTorch [out, in] weight layout."""

    def __init__(self, d: int, ffn: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(ffn, d))
        self.b1 = nn.Parameter(torch.empty(ffn))
        self.w2 = nn.Parameter(torch.empty(d, ffn))
        self.b2 = nn.Parameter(torch.empty(d))

    def forward(self, x):
        return moe_ref.gelu_mlp(x, self.w1, self.b1, self.w2, self.b2)


class Router(nn.Module):
    """Gate: logits = W_r x + b_r. Replicated on every rank (a dense parameter: averaged over dense DP)."""

    def __init__(self, d: int, n_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_experts, d))
        self.bias = nn.Parameter(torch.empty(n_experts))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class MoELayer(nn.Module):
    def __init__(self, cfg: MoEConfig, groups, impl, opts: RouteOpts):
        super().__init__()
        self.cfg, self.groups, self.impl, self.opts = cfg, groups, impl, opts
        ep = groups.ep if groups is not None else 1
        ep_rank = groups.ep_rank if groups is not None else 0
        assert cfg.n_experts % ep == 0, "E must be divisible by EP"
        self.e_loc = cfg.n_experts // ep
        self.owned = list(range(ep_rank * self.e_loc, (ep_rank + 1) * self.e_loc))
        self.router = Router(cfg.d_model, cfg.n_experts)
        self.experts = nn.ModuleDict({str(e): Expert(cfg.d_model, cfg.ffn) for e in self.owned})
        self.mark = None          # bench_skew sets a callable(name) that records a CUDA event
        self.stats: dict = {}

    def _mark(self, name: str) -> None:
        if self.mark is not None:
            self.mark(name)

    def _local_experts(self):
        return [self.experts[str(e)] for e in self.owned]

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        cfg, g, impl, opts = self.cfg, self.groups, self.impl, self.opts
        E, T = cfg.n_experts, x.shape[0]
        logits = self.router(x)
        if opts.forced_idx is not None:
            forced = opts.forced_idx.to(token_ids.device)[token_ids]
        else:
            forced = forced_experts(opts.pattern, token_ids, E, opts.n_tokens_global)
        idx, w, probs = impl.top1_route(logits, forced)
        counts = torch.bincount(idx, minlength=E)

        # Load balancing terms (GUIDE 2.3), attached to w so the layer output is unchanged.
        aux = z = None
        if opts.aux_scope != "off" and torch.is_grad_enabled():
            counts_for_f, t_for_f = counts, T
            if opts.aux_scope == "global" and g is not None and g.world > 1:
                counts_for_f = counts.clone()
                dist.all_reduce(counts_for_f, group=g.dp_group)        # E integers: global f_i on every rank
                t_for_f = T * g.world
            aux = moe_ref.aux_loss(probs, counts_for_f, t_for_f, E)
            z = moe_ref.z_loss(logits)
            w = moe_ref.AuxLossAutoScaler.apply(w, cfg.aux_coeff * aux + cfg.z_coeff * z)
        self._mark("route")

        keep = None
        if opts.capacity is not None:
            assert g is None or g.ep == 1, "capacity is a one-GPU experiment here (route_stats, check_local)"
            keep = moe_ref.capacity_keep_mask(idx, E, opts.capacity, opts.drop_policy, w.detach())
            order, counts_all = impl.permute(torch.where(keep, idx, torch.full_like(idx, E)), E + 1)
            kept = counts_all[:E]
            order = order[: int(kept.sum())]           # dynamic size: a device -> host sync (Megatron does the same)
        else:
            order, kept = impl.permute(idx, E)
        x_sorted = x.index_select(0, order)

        if g is None or g.ep == 1:
            host_counts = kept.tolist()                 # SequentialMLP's tokens_per_expert.tolist(): host sync
            self._mark("counts")
            y_sorted = moe_ref.run_experts(x_sorted, host_counts, self._local_experts())
            self._mark("experts")
            send = recv = None
        else:
            y_sorted, send, recv = self._dispatch_compute_combine(x_sorted, kept)

        y = impl.unpermute(y_sorted, order, T) * w.unsqueeze(1)
        self._mark("unpermute")
        self.stats = {"idx": idx.detach(), "counts": counts.detach(), "probs": probs.detach(),
                      "logits": logits.detach(), "aux": None if aux is None else aux.detach(),
                      "z": None if z is None else z.detach(), "keep": None if keep is None else keep.detach(),
                      "send": send, "recv": recv}
        return y

    def _dispatch_compute_combine(self, x_sorted: torch.Tensor, counts: torch.Tensor):
        g, impl, e_loc = self.groups, self.impl, self.e_loc
        ep = g.ep
        send = counts.view(ep, e_loc)                       # experts are contiguous per EP rank
        recv = impl.exchange_counts(send, g.ep_group)        # [src EP rank, my local expert]
        # ONE device -> host copy for every size the host needs. The CPU blocks here until the router, the sort and
        # the counts all_to_all have finished on the GPU: the dispatch cannot be enqueued without these numbers.
        host = torch.cat([send.sum(1), recv.sum(1), recv.reshape(-1)]).tolist()
        in_splits, out_splits, recv_chunks = host[:ep], host[ep:2 * ep], host[2 * ep:]
        self._mark("counts")
        if self.opts.validate:
            moe_ref.validate_splits_collective(in_splits, out_splits, x_sorted.shape[0], g.ep_group)
        x_recv = impl.AllToAllV.apply(x_sorted, out_splits, in_splits, g.ep_group)     # dispatch
        self._mark("dispatch")
        experts = self._local_experts()
        if e_loc > 1:                                        # chunks arrive (source, local expert): regroup
            by_expert, back = moe_ref.chunk_orders(ep, e_loc)
            x_e = moe_ref.sort_chunks(x_recv, recv_chunks, by_expert)
            per_expert = [sum(recv_chunks[s * e_loc + le] for s in range(ep)) for le in range(e_loc)]
            y_e = moe_ref.run_experts(x_e, per_expert, experts)
            y_recv = moe_ref.sort_chunks(y_e, [recv_chunks[i] for i in by_expert], back)
        else:
            y_recv = moe_ref.run_experts(x_recv, [sum(out_splits)], experts)
        self._mark("experts")
        y_sorted = impl.AllToAllV.apply(y_recv, in_splits, out_splits, g.ep_group)     # combine
        self._mark("combine")
        return y_sorted, in_splits, out_splits


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        with sdpa_kernel(SDPBackend.MATH):                  # plain matmul + softmax: repeatable (Day 01)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(b, s, d))


class Block(nn.Module):
    def __init__(self, cfg: MoEConfig, groups, impl, opts):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.moe = MoELayer(cfg, groups, impl, opts)

    def forward(self, h, token_ids):
        h = h + self.attn(self.ln1(h))
        b, s, d = h.shape
        return h + self.moe(self.ln2(h).reshape(b * s, d), token_ids).view(b, s, d)   # dropped token: + 0


class MoETransformer(nn.Module):
    def __init__(self, cfg: MoEConfig, groups, impl, opts):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, groups, impl, opts) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab)
        self.h0 = None

    def forward(self, tokens: torch.Tensor, token_offset: int) -> torch.Tensor:
        b, s = tokens.shape
        token_ids = token_offset + torch.arange(b * s, device=tokens.device)
        h = self.tok_emb(tokens) + self.pos_emb(torch.arange(s, device=tokens.device))
        if h.requires_grad:
            h.retain_grad()          # the "input gradient" compared against the baseline
        self.h0 = h
        for blk in self.blocks:
            h = blk(h, token_ids)
        return self.head(self.norm(h))

    def moe_layers(self):
        return [blk.moe for blk in self.blocks]


def build_model(cfg: MoEConfig, groups, impl, opts: RouteOpts, device, dtype=torch.float32) -> MoETransformer:
    """Same full weights on every rank and on the baseline; a rank creates only its own experts."""
    model = MoETransformer(cfg, groups, impl, opts).to(dtype=dtype)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(init_value(name, tuple(p.shape), cfg.seed, torch.float64))   # float64 draw, cast once
    model = model.to(device=device)
    for name, p in model.named_parameters():
        if ".experts." in name:
            p.expert_parallel = True     # Megatron: param.allreduce = False -> expert-data-parallel buffer
    return model


def expert_param_name(layer: int, expert: int, leaf: str) -> str:
    return f"blocks.{layer}.moe.experts.{expert}.{leaf}"
