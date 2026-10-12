from __future__ import annotations
from typing_extensions import override
import os
import torch
from ..ext import exllamav3_ext as ext

# Kernel-fused issue/collect for decode-size split jobs (EXL3_MOE_SPLIT_FUSED=0 restores the
# cudaMemcpyAsync path for A/B testing)
_split_fused = os.environ.get("EXL3_MOE_SPLIT_FUSED", "1") != "0"

# EXL3_SPLIT_PROF=1: CUDA-event brackets around the fused issue enqueue and the collect
# wait+readback, reported as stream-time percentiles every ~2048 brackets. The collect
# bracket is the true exposed stall: it runs after the GPU's own expert work, so any time
# there is worker lateness the GPU could not hide.
_split_prof = bool(os.environ.get("EXL3_SPLIT_PROF"))
_sprof = {"issue": [], "wait": []}

def _sprof_wrap(kind, layer, fn):
    ev0 = torch.cuda.Event(enable_timing = True)
    ev1 = torch.cuda.Event(enable_timing = True)
    ev0.record()
    r = fn()
    ev1.record()
    recs = _sprof[kind]
    recs.append((layer, ev0, ev1))
    if kind == "wait" and len(recs) >= 2048:
        for k, rs in _sprof.items():
            ts = [(l, a.elapsed_time(b)) for l, a, b in rs if b.query()]
            if not ts:
                continue
            vals = sorted(t for _, t in ts)
            per_layer = {}
            for l, t in ts:
                s = per_layer.setdefault(l, [0.0, 0])
                s[0] += t; s[1] += 1
            worst = sorted(per_layer.items(), key = lambda kv: -kv[1][0] / kv[1][1])[:4]
            print(f" -- split prof [{k}] ({len(vals)} brackets, stream ms): "
                  f"med {vals[len(vals) // 2]:.3f} p90 {vals[int(len(vals) * 0.9)]:.3f} "
                  f"max {vals[-1]:.3f} | worst layers "
                  + " ".join(f"L{l}:{s / n:.3f}" for l, (s, n) in worst), flush = True)
            _sprof[k] = []
    return r

"""
Forward-path hooks are cpu_split_submit / cpu_offload_forward / cpu_split_combine, the
load-path hooks cpu_maybe_offload_load / cpu_maybe_split_load / cpu_post_load, plus
cpu_unload. The worker process itself lives in model/moe_cpu_host.py.
"""


def run_pending_swap_sweeps(infer_params):
    """Run any pending dynamic-placement sweep (see BlockSparseMLP._split_swap_tick). Called
    by the generator when its job queue drains, so placement only ever changes between
    generations; safe no-op otherwise."""
    if not getattr(infer_params, "moe_cpu_swap_pending", False):
        return
    infer_params.moe_cpu_swap_pending = False
    reg = getattr(infer_params, "moe_cpu_swap_modules", None)
    if not reg:
        return
    reg[0]._split_sweep_layer_reset()
    # Quiesce every device holding a split layer, not just the current one: with a
    # multi-GPU layer split, worker jobs are issued from each device's stream, and their
    # completion is only guaranteed by the collect memop waits on those streams. A
    # single-device synchronize leaves other devices' jobs in flight while the sweep
    # rewrites arena slots. After the sync, every issued job's collect has completed, so
    # the ring is drained and the worker idle by construction.
    for d in sorted({m.device.index for m in reg if m.device is not None}):
        torch.cuda.synchronize(d)
    if os.environ.get("EXL3_MOE_CPU_SWAP_VERIFY"):
        for h in {m.cpu_host for m in reg}:
            assert int(h.v_jobs_head[0]) == int(h.v_jobs_tail[0]), \
                "swap sweep: job ring not drained after all-device sync"
    budget = int(os.environ.get("EXL3_MOE_CPU_SWAP_MAX", 64))
    per_layer = int(os.environ.get("EXL3_MOE_CPU_SWAP_PER_LAYER", 0))
    total = 0
    for m in reg:
        # Always visit every layer so its histogram is snapshotted and decayed. A
        # per-layer cap avoids letting the first few layers consume the entire global
        # promotion budget, which otherwise starves later layers indefinitely.
        layer_budget = max(budget, 0)
        if per_layer:
            layer_budget = min(layer_budget, per_layer)
        n = m._split_sweep_layer(layer_budget)
        budget -= n
        total += n
    hist_out = os.environ.get("EXL3_MOE_CPU_SWAP_HIST_OUT")
    if hist_out:
        import json
        payloads = {
            hist_out: {
                m.key: m._split_last_hist
                for m in reg
                if getattr(m, "_split_last_hist", None) is not None
            },
        }
        if os.environ.get("EXL3_MOE_CPU_SPLIT_PHASE_HIST"):
            payloads[hist_out + ".prefill.json"] = {
                m.key: m._split_last_hist_prefill
                for m in reg
                if getattr(m, "_split_last_hist_prefill", None) is not None
            }
            payloads[hist_out + ".decode.json"] = {
                m.key: m._split_last_hist_decode
                for m in reg
                if getattr(m, "_split_last_hist_decode", None) is not None
            }
        if any(getattr(m, "_split_last_lru", None) is not None for m in reg):
            payloads[hist_out + ".lru.json"] = {
                m.key: m._split_last_lru
                for m in reg
                if getattr(m, "_split_last_lru", None) is not None
            }
        for path, payload in payloads.items():
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, sort_keys = True)
                f.write("\n")
            os.replace(tmp, path)
    if os.environ.get("EXL3_MOE_CPU_SWAP_DEBUG"):
        hits = sum(getattr(m, "_split_last_gpu_hits", 0.0) for m in reg)
        mass = sum(getattr(m, "_split_last_total_hits", 0.0) for m in reg)
        ratio = hits / mass if mass else 0.0
        print(f" -- expert swap sweep: {total} swaps, pre-sweep GPU hit "
              f"{ratio:.4f} ({hits:.0f}/{mass:.0f})", flush = True)


class BlockSparseMLP_CPU:

    def _cpu_init_state(self):
        self.cpu_offload = False
        self.cpu_split_first = None   # split offload: first CPU-resident (tail) expert index
        self._split_map = None        # dynamic placement: router id -> physical slot
        self._split_hist = None
        self._split_hist_prefill = None
        self._split_hist_decode = None
        self._split_phase_selcpu_t = None
        # Optional mirror-cache placement. The GPU keeps [0, fixed_first) permanently
        # resident and [fixed_first, cpu_split_first) as replaceable mirror slots. The CPU
        # worker owns every expert at and above fixed_first, including the mirrored ones,
        # so an eviction is a map update rather than a GPU -> CPU copy.
        self._split_fixed_first = None
        self._split_mirror_slots = 0
        self._split_cpu_physical = None
        self._split_lru_sims = []
        self._split_last_lru = None

    def cpu_maybe_offload_load(self, device, **kwargs) -> bool:
        """Whole-layer offload claim: when the budget allows and the layer is eligible,
        register the routed experts with the CPU worker instead of loading them; the caller
        skips its GPU load entirely on True."""
        ip = self.config.infer_params
        comp = getattr(ip, "moe_cpu_component", "text")
        budget = getattr(ip, "moe_cpu_offload", 0) if comp == "text" \
            else getattr(ip, "draft_moe_cpu_offload", 0)
        if budget:
            assert not getattr(ip, "moe_cpu_split", 0), \
                "moe_cpu_split and moe_cpu_offload are mutually exclusive: the split offloads " \
                "a slice of every eligible layer's experts, whole-layer offload takes entire " \
                "layers — pick one"
        if (
            budget > 0 and
            ip.moe_cpu_offload_assigned.get(comp, 0) < budget and
            device is not None and torch.device(device).type == "cuda" and
            (self.num_local_experts is None or self.num_local_experts == self.num_experts) and
            (self.activation_fn in ("silu", "gelu", "swiglu_oai") if self.gated else self.activation_fn == "relu2")
        ):
            if self.load_cpu_offload(device, **kwargs):
                ip.moe_cpu_offload_assigned[comp] = ip.moe_cpu_offload_assigned.get(comp, 0) + 1
                return True

        return False

    def cpu_maybe_split_load(self, device, **kwargs):
        """Per-layer expert split registration (EXL3_MOE_CPU_SPLIT = tail experts per layer
        on the CPU worker): shrinks the module to its GPU slice; the caller's normal load
        then loads that slice."""
        ip = self.config.infer_params
        # Registration shrinks the module to its GPU slice, then the normal load below loads
        # that slice. infer_params.moe_cpu_split is authoritative (the EXL3_MOE_CPU_SPLIT env
        # is its construction-time default)
        split_k = int(getattr(ip, "moe_cpu_split", 0))
        if split_k:
            assert not getattr(ip, "moe_cpu_offload", 0) and not getattr(ip, "draft_moe_cpu_offload", 0), \
                "moe_cpu_split and moe_cpu_offload are mutually exclusive: the split offloads " \
                "a slice of every eligible layer's experts, whole-layer offload takes entire " \
                "layers — pick one"
        split_layers = int(os.environ.get("EXL3_MOE_CPU_SPLIT_LAYERS", 0))
        if split_layers and getattr(ip, "moe_cpu_split_assigned", 0) >= split_layers:
            split_k = 0
        if (
            0 < split_k < self.num_experts and
            self.cpu_split_first is None and
            device is not None and torch.device(device).type == "cuda" and
            self.num_local_experts == self.num_experts and self.routing_first is None and
            (self.activation_fn in ("silu", "gelu", "swiglu_oai") if self.gated else self.activation_fn == "relu2")
        ):
            if self.load_cpu_split(device, split_k, **kwargs):
                ip.moe_cpu_split_assigned = getattr(ip, "moe_cpu_split_assigned", 0) + 1

    def cpu_post_load(self):
        """After the module's tensors load on a CUDA device, set up the router-id to
        physical-slot translation used by static or dynamic expert placement."""
        # Frequency-guided static placement physically rearranges only the expert weight
        # lists. Keep the router, correction bias and per-expert scales in their trained
        # order, then translate each original router id to the corresponding physical slot
        # immediately after routing. This is the same mechanism used by dynamic placement
        # and avoids changing the model's logits or routing numerics at load time.
        if self.cpu_split_first is not None and self._split_mirror_slots:
            dev = torch.device(self.device)
            perm = self._split_perm or list(range(self.num_experts))
            perm_t = torch.tensor(perm, dtype = torch.long)
            assert perm_t.numel() == self.num_experts and \
                   torch.unique(perm_t).numel() == self.num_experts, \
                   "mirror expert placement must be a permutation"
            fixed = self._split_fixed_first
            gpu_first = self.cpu_split_first
            # The worker's local slot for physical rank r is r - fixed. Physical ids
            # begin at gpu_first because the split kernels subtract cpu_split_first.
            cpu_physical = torch.full((self.num_experts,), -1, dtype = torch.long)
            placement = torch.empty(self.num_experts, dtype = torch.long)
            for rank, router_id in enumerate(perm):
                if rank < gpu_first:
                    placement[router_id] = rank
                else:
                    placement[router_id] = gpu_first + rank - fixed
                if rank >= fixed:
                    cpu_physical[router_id] = gpu_first + rank - fixed
            self._split_map = placement.to(dev)
            self._split_cpu_physical = cpu_physical
            self._split_hist = torch.zeros(self.num_experts, dtype = torch.float, device = dev)
            self._split_init_phase_hist(dev)
            self._split_selcpu_t = None
            self._split_phase_selcpu_t = None
            if self._split_dynamic:
                self._swap_tick_count = 0
                ip2 = self.config.infer_params
                reg = getattr(ip2, "moe_cpu_swap_modules", None)
                if reg is None:
                    reg = []
                    ip2.moe_cpu_swap_modules = reg
                reg.append(self)
        elif self.cpu_split_first is not None and self._split_perm is not None:
            dev = torch.device(self.device)
            perm_t = torch.tensor(self._split_perm, dtype = torch.long, device = dev)
            assert perm_t.numel() == self.num_experts and \
                   torch.unique(perm_t).numel() == self.num_experts, \
                   "static expert placement must be a permutation"
            self._split_map = torch.empty(self.num_experts, dtype = torch.long, device = dev)
            self._split_map[perm_t] = torch.arange(
                self.num_experts, dtype = torch.long, device = dev)
            self._split_hist = torch.zeros(self.num_experts, dtype = torch.float, device = dev)
            self._split_init_phase_hist(dev)
            self._split_selcpu_t = None
            self._split_phase_selcpu_t = None
            # Static placements still need a queue-drained snapshot path for offline
            # routing traces. Register them only when an explicit histogram output is
            # requested; the sweep budget can stay at zero, so this never moves experts.
            if os.environ.get("EXL3_MOE_CPU_SWAP_HIST_OUT"):
                self._split_register_sweep_tick()
        # Dynamic placement state: router->physical-slot map (identity start), decayed
        # selection counts, and the shared module registry the sweep walks. The first
        # registered module owns the step counter
        elif self.cpu_split_first is not None and getattr(self, "_split_dynamic", False):
            dev = torch.device(self.device)
            self._split_map = torch.arange(self.num_experts, dtype = torch.long, device = dev)
            self._split_hist = torch.zeros(self.num_experts, dtype = torch.float, device = dev)
            self._split_init_phase_hist(dev)
            self._split_selcpu_t = None
            self._split_phase_selcpu_t = None
            self._split_register_sweep_tick()

        if self.cpu_split_first is not None and self._split_map is not None:
            self._split_init_lru_sim(torch.device(self.device))

    def _split_register_sweep_tick(self):
        self._swap_tick_count = 0
        ip = self.config.infer_params
        reg = getattr(ip, "moe_cpu_swap_modules", None)
        if reg is None:
            reg = []
            ip.moe_cpu_swap_modules = reg
        if self not in reg:
            reg.append(self)

    def cpu_unload(self):
        if self.cpu_split_first is not None:
            host = getattr(self, "cpu_host", None)
            if host is not None:
                host.unregister()
            (self.gates, self.ups, self.downs, self.modules,
             self.num_local_experts, self.routing_first, self.routing_last) = self._split_saved
            self._split_saved = None
            self.cpu_split_first = None
            if self._split_map is not None:
                reg = getattr(self.config.infer_params, "moe_cpu_swap_modules", None)
                if reg is not None and self in reg:
                    reg.remove(self)
                self._split_map = None
                self._split_hist = None
                self._split_hist_prefill = None
                self._split_hist_decode = None
            self._split_lru_sims = []
            self._split_last_lru = None
            self._split_fixed_first = None
            self._split_mirror_slots = 0
            self._split_cpu_physical = None
        if self.cpu_offload:
            host = getattr(self, "cpu_host", None)
            if host is not None:
                host.unregister()
            # Release this layer's slot in the component's offload budget so a reload of the
            # same config can claim it again
            asn = self.config.infer_params.moe_cpu_offload_assigned
            comp = getattr(self, "cpu_component", "text")
            if isinstance(asn, dict) and asn.get(comp, 0) > 0:
                asn[comp] -= 1
        self.cpu_offload = False

    def cpu_split_submit(self, y, bsz, selected_experts, routing_weights):
        """Hand the tail experts' share of the routed sum to the worker. Returns
        (cpu_partial, cpu_pending): decode-size batches use the two-phase issue/collect so
        the flag waits land AFTER the caller's GPU expert work (cpu_split_combine collects);
        big prefill batches take the single-phase streamed path, which overlaps internally.
        Expert ids ship in the worker's local range with -1 sentinels for GPU-resident
        picks."""
        # This must run before moe_split_map translates selected_experts in place.
        self._split_record_lru_sim(selected_experts, bsz)
        if self._split_map is not None and (
            self._split_dynamic or os.environ.get("EXL3_MOE_CPU_SWAP_HIST_OUT")
        ):
            self._split_swap_tick()
        if bsz < self.cpu_host.stream_min_rows:
            # Fused fast path: one kernel does the map translate + staging writes straight
            # into the pinned slot, skipping the whole activation payload (and, downstream,
            # the worker compute and the readback) when no CPU expert was selected. Replaces
            # three cudaMemcpyAsync launches, the int32 cast and the pad per layer per step
            if _split_fused:
                self._split_record_phase_hist(selected_experts, bsz)
                call = lambda: self.cpu_host.submit_issue_fused(
                    self.cpu_layer_idx, y, selected_experts, routing_weights,
                    self._split_map,
                    self._split_hist if self._split_map is not None else None,
                    self.cpu_split_first)
                pending = _sprof_wrap("issue", self.cpu_layer_idx, call) \
                    if _split_prof else call()
                if pending is not None:
                    return None, ("fused", pending)
            sel_cpu = self._split_translate(selected_experts, bsz)
            return None, self.cpu_host.submit_issue(
                self.cpu_layer_idx, y, sel_cpu, routing_weights)
        sel_cpu = self._split_translate(selected_experts, bsz)
        return self.cpu_host.submit_prefill(
            self.cpu_layer_idx, y, sel_cpu, routing_weights), None

    def _split_init_phase_hist(self, dev):
        if os.environ.get("EXL3_MOE_CPU_SPLIT_PHASE_HIST"):
            self._split_hist_prefill = torch.zeros(
                self.num_experts, dtype = torch.float, device = dev)
            self._split_hist_decode = torch.zeros(
                self.num_experts, dtype = torch.float, device = dev)
        else:
            self._split_hist_prefill = None
            self._split_hist_decode = None

    def _split_init_lru_sim(self, dev):
        """Initialize optional, non-mutating decode LRU simulations. Capacities are a
        comma-separated EXL3_MOE_CPU_LRU_SIM list. Each simulated cache starts with the
        current frequency-ranked GPU placement, then tracks exact per-token LRU churn.
        Larger-than-current capacities start with empty extra slots."""
        spec = os.environ.get("EXL3_MOE_CPU_LRU_SIM", "").strip()
        if not spec:
            return
        capacities = sorted({int(x) for x in spec.split(",") if x.strip()})
        if not capacities or capacities[0] <= 0 or capacities[-1] > self.num_experts:
            raise ValueError(
                f"EXL3_MOE_CPU_LRU_SIM capacities must be in [1, {self.num_experts}]")
        placement = self._split_map.detach().cpu()
        initial = sorted(
            (int(p), r) for r, p in enumerate(placement.tolist())
            if int(p) < self.cpu_split_first)
        initial = [r for _, r in initial]
        for capacity in capacities:
            seeds = initial[:capacity]
            cache_map = torch.full(
                (self.num_experts,), -1, dtype = torch.long, device = dev)
            slot_experts = torch.full(
                (capacity,), -1, dtype = torch.long, device = dev)
            slot_ages = torch.zeros(capacity, dtype = torch.long, device = dev)
            if seeds:
                ids = torch.tensor(seeds, dtype = torch.long, device = dev)
                slots = torch.arange(len(seeds), dtype = torch.long, device = dev)
                cache_map[ids] = slots
                slot_experts[:len(seeds)] = ids
                # Physical order follows the frequency-ranked placement. Make the least
                # frequent initial tenant the first eviction victim.
                slot_ages[:len(seeds)] = torch.arange(
                    len(seeds), 0, -1, dtype = torch.long, device = dev)
            self._split_lru_sims.append(dict(
                capacity = capacity,
                cache_map = cache_map,
                slot_experts = slot_experts,
                slot_ages = slot_ages,
                clock = torch.tensor([len(seeds)], dtype = torch.long, device = dev),
                counters = torch.zeros(2, dtype = torch.long, device = dev),
            ))

    def _split_record_lru_sim(self, selected_experts, bsz):
        # Decode only. Prompt processing can contain thousands of rows and is a different
        # cache problem; excluding it keeps this exact scalar reference probe negligible.
        if bsz != 1:
            return
        for sim in self._split_lru_sims:
            ext.moe_lru_sim(
                selected_experts.view(-1), sim["cache_map"], sim["slot_experts"],
                sim["slot_ages"], sim["clock"], sim["counters"])

    def _split_record_phase_hist(self, selected_experts, bsz):
        if self._split_hist_prefill is None:
            return
        phase_hist = self._split_hist_decode if bsz == 1 else self._split_hist_prefill
        ext.moe_split_hist(selected_experts.view(-1), phase_hist)

    def _split_translate(self, selected_experts, bsz = None):
        """Copy-path selection translate: dynamic placement runs the map kernel (hit counts,
        in-place physical-slot translate -- the bsz-1 routing statics are read by captured
        graphs at baked addresses -- and the worker-side selection with -1 sentinels for
        GPU-resident picks); the static split is a tail offset."""
        if self._split_map is None:
            return (selected_experts - self.cpu_split_first).clamp_min(-1)
        n = selected_experts.numel()
        buf = self._split_selcpu_t
        if buf is None or buf.numel() < n:
            buf = self._split_selcpu_t = torch.empty(
                max(n, 64), dtype = torch.long, device = selected_experts.device)
        sel_cpu = buf[:n].view(selected_experts.shape)
        ext.moe_split_map(
            selected_experts.view(-1), self._split_map, self._split_hist,
            buf[:n], self.cpu_split_first)
        # Optional trace-only second pass keeps exact prefill and decode routing counts.
        # The normal combined histogram and translated selection stay authoritative; the
        # extra tiny map kernel is enabled only for offline workload characterization.
        if self._split_hist_prefill is not None:
            self._split_record_phase_hist(selected_experts, bsz)
        return sel_cpu

    def cpu_offload_forward(self, x, y, selected_experts, routing_weights, params):
        """Whole-layer offload: the routed sum comes entirely from the worker. The autosplit
        measuring forward only observes VRAM allocation, which the CPU compute cannot
        affect, so it skips the (slow, full-chunk) host pass and just allocates the
        output."""
        if params.get("autosplit_measure"):
            return torch.zeros_like(y, dtype = torch.float).reshape(x.shape)
        return self.cpu_host.submit_prefill(
            self.cpu_layer_idx, y, selected_experts, routing_weights
        ).reshape(x.shape)

    def cpu_split_combine(self, final_hidden_states, cpu_partial, cpu_pending, x):
        """Fold the CPU tail partial into the routed sum (stream-ordered: the collect
        enqueues a flag wait ahead of the readback, so the add consumes the worker's output
        exactly when it is ready). Fused handles add in place straight from the pinned slot
        (or skip the readback entirely for a job with no CPU-resident picks)."""
        if cpu_pending is not None:
            if isinstance(cpu_pending, tuple) and cpu_pending[0] == "fused":
                f2 = final_hidden_states.view(-1, final_hidden_states.shape[-1])
                if _split_prof:
                    _sprof_wrap("wait", self.cpu_layer_idx,
                                lambda: self.cpu_host.submit_collect_fused(cpu_pending[1], f2))
                else:
                    self.cpu_host.submit_collect_fused(cpu_pending[1], f2)
                return final_hidden_states
            cpu_partial = self.cpu_host.submit_collect(cpu_pending)
        if cpu_partial is not None:
            final_hidden_states = final_hidden_states + cpu_partial.view(x.shape)
        return final_hidden_states

    @override
    def can_defer_load(self):
        # Keep frequency-guided placement on the conservative synchronous load path. The
        # physical expert lists are rewritten before load and the inverse map is installed
        # immediately afterwards; avoiding deferred fills here makes that ordering explicit.
        if (
            int(os.environ.get("EXL3_MOE_CPU_SPLIT", 0)) > 0 and
            os.environ.get("EXL3_MOE_CPU_SPLIT_STATS")
        ):
            return False
        return super().can_defer_load()

    def load_cpu_offload(self, device: torch.Device, **kwargs) -> bool:
        """
        Experimental CPU expert offload: register the layer with the persistent CPU MoE worker
        (which loads the expert weights itself, concurrently with GPU loading) and load
        everything else (router, norms, shared experts) on the GPU as usual. Eligibility here
        uses header metadata only; the parent never fetches expert data. Returns False without
        side effects when the layer is ineligible (non-mul1 codebook, K > 8, or mixed per-expert
        biases), in which case the caller falls back to the normal path.
        """
        stc = self.config.stc
        cpu = torch.device("cpu")
        experts = self.gates + self.ups + self.downs

        # Eligibility probe on the first expert of each projection before fetching bulk data
        probe = ([self.gates[0]] if self.gated else []) + [self.ups[0], self.downs[0]]
        for l in probe:
            if stc.get_tensor(l.key + ".mul1", cpu, optional = True) is None:
                print(f" !! {self.key}: experts are not mul1, CPU offload skipped")
                return False

        def hdr_shape(l):
            return stc.list_tensors(l.key)[l.key + ".trellis"]["shape"]
        for l in probe:
            if hdr_shape(l)[-1] // 16 > 8:
                print(f" !! {self.key}: K > 8, CPU offload skipped")
                return False
        def bias_keys(ls):
            has = [(l.key + ".bias") in stc.tensor_file_map for l in ls]
            if any(has) and not all(has):
                print(f" !! {self.key}: mixed expert biases, CPU offload skipped")
                return None
            return all(has)
        checks = [bias_keys(ls) for ls in ([self.gates] if self.gated else []) + [self.ups, self.downs]]
        if any(c is None for c in checks):
            return False

        self.device = torch.device(device)
        expert_set = set(experts)
        for module in self.modules:
            if module not in expert_set:
                module.load(device, **kwargs)
        if self.e_score_correction_bias_key:
            for k in [self.e_score_correction_bias_key, "gate.e_score_correction_bias"]:
                # Loaded in its checkpoint precision: GLM-5.2's bias sits near 34.0 where
                # fp16 resolution (0.03) would swamp the inter-expert sigmoid score gaps. The
                # kernels get a centered fp16 copy (see RoutingCFG.e_score_bias_h)
                # no_defer: the fp32 working copy below is derived at load time, which a
                # deferred (unfilled) tensor would corrupt
                esb = self.config.stc.get_tensor(
                    f"{self.key}.{k}", self.device, optional = True, allow_bf16 = True,
                    no_defer = True)
                if esb is not None:
                    self.e_score_correction_bias = esb if esb.dtype == torch.half else esb.float()
                    break
        if self.per_expert_scale_key:
            self.per_expert_scale = self.config.stc.get_tensor(
                f"{self.key}.{self.per_expert_scale_key}", self.device, optional = True, allow_bf16 = True)
        if self.tid2eid_key:
            self.tid2eid = self.config.stc.get_tensor(
                f"{self.key}.{self.tid2eid_key}", self.device, no_defer = True)
        self.load_routing(**kwargs)

        from ..model.moe_cpu_host import MoeCpuHost
        # One worker per component: an MTP head shares the config but loads after the main
        # model's worker has started, so it gets its own child (which loads only its own
        # layers from the same checkpoint)
        comp = getattr(self.config.infer_params, "moe_cpu_component", "text")
        hosts = getattr(self.config, "moe_cpu_hosts", None)
        if hosts is None:
            hosts = {}
            self.config.moe_cpu_hosts = hosts
        host = hosts.get(comp)
        if host is None:
            host = MoeCpuHost(self.config)
            hosts[comp] = host
        self.cpu_host = host
        self.cpu_component = comp
        def dims_of(l):
            s = stc.list_tensors(l.key)[l.key + ".trellis"]["shape"]
            return (s[0] * 16, s[1] * 16, s[2] // 16)
        gd = dims_of(self.gates[0]) if self.gated else None
        ud = dims_of(self.ups[0])
        dd = dims_of(self.downs[0])
        hi, ho = ud[0], dd[1]

        # Small per-expert tensors resident on the GPU for the streamed-prefill dequant path
        # (lists, not stacks: the fetches may be deferred and fill in place)
        def fetch_aux(ls, suffix, optional = False):
            out = [stc.get_tensor(l.key + suffix, self.device, optional = optional,
                                  float2half = True) for l in ls]
            return out if not optional or out[0] is not None else None
        aux = dict(
            suh_u = fetch_aux(self.ups, ".suh"), svh_u = fetch_aux(self.ups, ".svh"),
            suh_d = fetch_aux(self.downs, ".suh"), svh_d = fetch_aux(self.downs, ".svh"),
            bias_u = fetch_aux(self.ups, ".bias", True),
            bias_d = fetch_aux(self.downs, ".bias", True),
        )
        if self.gated:
            aux["suh_g"] = fetch_aux(self.gates, ".suh")
            aux["svh_g"] = fetch_aux(self.gates, ".svh")
            aux["bias_g"] = fetch_aux(self.gates, ".bias", True)

        self.cpu_layer_idx = host.register_layer(
            self.key,
            [l.key for l in self.gates] if self.gated else [],
            [l.key for l in self.ups],
            [l.key for l in self.downs],
            {"silu": 0, "gelu": 1, "relu2": 2, "swiglu_oai": 3}[self.activation_fn],
            float(self.act_limit or 0.0),
            hi, ho, self.num_experts_per_tok,
            proj_dims = dict(g = gd, u = ud, d = dd),
            aux = aux,
        )
        self.cpu_offload = True
        print(f" -- CPU-offloaded experts (worker): {self.key}")
        return True


    def load_cpu_split(self, device: torch.Device, split_k: int, **kwargs) -> bool:
        """
        Experimental per-layer expert split: register the TAIL split_k experts with the CPU MoE
        worker and shrink this module to a GPU-resident [0, E - split_k) slice (the existing
        expert-interval machinery masks the tail out of every GPU path, exactly like a TP
        expert shard). forward() then submits the tail partial to the worker right after
        routing and adds it to the GPU partial, so the CPU GEMMs overlap the layer's own GPU
        expert compute instead of serializing a whole layer. Registration only. The caller
        proceeds with the normal load, which now loads just the GPU slice.
        """
        stc = self.config.stc
        cpu = torch.device("cpu")
        # In ordinary split mode ``first`` is both the GPU/CPU boundary and the first
        # worker expert. Mirror mode decouples them: the worker still owns [first, E),
        # while the GPU also holds ``mirror_slots`` replaceable copies from that range.
        first = self.num_experts - split_k
        mirror_slots = int(os.environ.get("EXL3_MOE_CPU_MIRROR_SLOTS", 0))
        if mirror_slots < 0 or mirror_slots > split_k:
            raise ValueError(
                f"EXL3_MOE_CPU_MIRROR_SLOTS={mirror_slots} must be in [0, {split_k}]")
        if mirror_slots and self.tid2eid_key:
            print(f" !! {self.key}: tid2eid remap present, mirror cache disabled")
            mirror_slots = 0
        gpu_first = first + mirror_slots

        # Same eligibility probes as the whole-layer path, on the tail experts
        probe = ([self.gates[first]] if self.gated else []) + [self.ups[first], self.downs[first]]
        for l in probe:
            if stc.get_tensor(l.key + ".mul1", cpu, optional = True) is None:
                print(f" !! {self.key}: experts are not mul1, CPU split skipped")
                return False
        def hdr_shape(l):
            return stc.list_tensors(l.key)[l.key + ".trellis"]["shape"]
        for l in probe:
            if hdr_shape(l)[-1] // 16 > 8:
                print(f" !! {self.key}: K > 8, CPU split skipped")
                return False
        def bias_keys(ls):
            has = [(l.key + ".bias") in stc.tensor_file_map for l in ls[first:]]
            if any(has) and not all(has):
                print(f" !! {self.key}: mixed expert biases, CPU split skipped")
                return None
            return all(has)
        checks = [bias_keys(ls) for ls in ([self.gates] if self.gated else []) + [self.ups, self.downs]]
        if any(c is None for c in checks):
            return False

        # Optional frequency-guided placement (EXL3_MOE_CPU_SPLIT_STATS = json of per-layer
        # selection counts, e.g. from a probe run): permute the expert order hot-to-cold so
        # the offloaded tail holds the least-selected experts. Router-side tensors remain in
        # their trained order; cpu_post_load installs an inverse map from original router id
        # to the physical hot-to-cold slot used by GPU pointer tables and the worker key list.
        # Measured on lfm2.5: the coldest 12/32 experts draw ~8% of selections vs 37.5%
        # uniform, cutting the CPU share per layer ~4.5x
        self._split_perm = None
        self._split_saved_lists = (self.gates, self.ups, self.downs)
        # Dynamic placement (default; EXL3_MOE_CPU_SWAP=0 disables). Ordinary swaps exchange
        # GPU and CPU tenants. Mirror mode instead keeps CPU masters for every non-fixed
        # expert and only replaces the GPU copy, matching the Qwen LRU cache architecture.
        self._split_dynamic = os.environ.get("EXL3_MOE_CPU_SWAP", "1") != "0" \
            and not self.tid2eid_key
        self._split_mirror_slots = mirror_slots
        self._split_fixed_first = first if mirror_slots else None
        stats_path = os.environ.get("EXL3_MOE_CPU_SPLIT_STATS")
        if stats_path and self._split_dynamic and not mirror_slots:
            # Static placement from a stats file only applies with dynamic swapping disabled
            print(f" !! {self.key}: EXL3_MOE_CPU_SPLIT_STATS ignored, set EXL3_MOE_CPU_SWAP=0 to use it")
            stats_path = None
        if stats_path and self.tid2eid_key:
            print(f" !! {self.key}: tid2eid remap present, tail placement unpermuted")
            stats_path = None
        if stats_path:
            import json
            counts = json.load(open(stats_path)).get(self.key)
            if counts is not None and len(counts) == self.num_experts:
                perm = sorted(range(self.num_experts), key = lambda e: -counts[e])
                self._split_perm = perm
                if self.gated:
                    self.gates = [self.gates[e] for e in perm]
                self.ups = [self.ups[e] for e in perm]
                self.downs = [self.downs[e] for e in perm]
            else:
                print(f" !! {self.key}: no routing stats for layer, tail placement unpermuted")

        self.device = torch.device(device)

        from ..model.moe_cpu_host import MoeCpuHost
        comp = getattr(self.config.infer_params, "moe_cpu_component", "text")
        hosts = getattr(self.config, "moe_cpu_hosts", None)
        if hosts is None:
            hosts = {}
            self.config.moe_cpu_hosts = hosts
        host = hosts.get(comp)
        if host is None:
            host = MoeCpuHost(self.config)
            hosts[comp] = host
        self.cpu_host = host
        self.cpu_component = comp

        def dims_of(l):
            s = stc.list_tensors(l.key)[l.key + ".trellis"]["shape"]
            return (s[0] * 16, s[1] * 16, s[2] // 16)
        gd = dims_of(self.gates[first]) if self.gated else None
        ud = dims_of(self.ups[first])
        dd = dims_of(self.downs[first])
        hi, ho = ud[0], dd[1]

        def fetch_aux(ls, suffix, optional = False):
            out = [stc.get_tensor(l.key + suffix, self.device, optional = optional,
                                  float2half = True) for l in ls[first:]]
            return out if not optional or out[0] is not None else None
        aux = dict(
            suh_u = fetch_aux(self.ups, ".suh"), svh_u = fetch_aux(self.ups, ".svh"),
            suh_d = fetch_aux(self.downs, ".suh"), svh_d = fetch_aux(self.downs, ".svh"),
            bias_u = fetch_aux(self.ups, ".bias", True),
            bias_d = fetch_aux(self.downs, ".bias", True),
        )
        if self.gated:
            aux["suh_g"] = fetch_aux(self.gates, ".suh")
            aux["svh_g"] = fetch_aux(self.gates, ".svh")
            aux["bias_g"] = fetch_aux(self.gates, ".bias", True)

        self.cpu_layer_idx = host.register_layer(
            self.key,
            [l.key for l in self.gates[first:]] if self.gated else [],
            [l.key for l in self.ups[first:]],
            [l.key for l in self.downs[first:]],
            {"silu": 0, "gelu": 1, "relu2": 2, "swiglu_oai": 3}[self.activation_fn],
            float(self.act_limit or 0.0),
            hi, ho, self.num_experts_per_tok,
            proj_dims = dict(g = gd, u = ud, d = dd),
            aux = aux,
        )

        # Shrink to the GPU slice. In mirror mode the initial top-ranked mirror experts stay
        # in the module as GPU copies even though the worker also registers their masters.
        # The remaining tail Linears leave the module tree entirely (never loaded); unload()
        # restores them so a reload can redo the split cleanly.
        tail = set((self.gates[gpu_first:] if self.gated else [])
                   + self.ups[gpu_first:] + self.downs[gpu_first:])
        self._split_saved = (*self._split_saved_lists, self.modules,
                            self.num_local_experts, self.routing_first, self.routing_last)
        del self._split_saved_lists
        self.gates = self.gates[:gpu_first] if self.gated else self.gates
        self.ups = self.ups[:gpu_first]
        self.downs = self.downs[:gpu_first]
        self.modules = [m for m in self.modules if m not in tail]
        self.num_local_experts = gpu_first
        self.routing_first = 0
        self.routing_last = gpu_first
        self.cpu_split_first = gpu_first
        mode = "dynamic" if self._split_dynamic else "static"
        if mirror_slots:
            mode += f", mirror={mirror_slots}, fixed={first}"
        print(f" -- CPU split experts (worker, {mode}): {self.key} "
              f"masters [{first}..{self.num_experts}), GPU [0..{gpu_first})")
        return True


    def _split_swap_tick(self):
        """Dynamic placement sweep trigger: the first registered module counts decode
        steps; every EXL3_MOE_CPU_SWAP_INTERVAL steps, quiesce (one full synchronize,
        every issued worker job's collect is already enqueued, so after the sync the ring is
        drained and the child idle) and let each split layer promote its hottest CPU experts
        into the slots of its coldest GPU experts."""
        ip = self.config.infer_params
        reg = ip.moe_cpu_swap_modules
        if reg[0] is not self:
            return
        self._swap_tick_count += 1
        interval = int(os.environ.get("EXL3_MOE_CPU_SWAP_INTERVAL", 128))
        if self._swap_tick_count < interval:
            return
        # A placement change is a small numeric step for every touched expert (the same
        # weights compute slightly differently on the two devices), so sweeping mid-stream
        # visibly derails a generation in progress. Mark the sweep pending and let the
        # generator run it from its queue-drained hook; fall back to sweeping inline only
        # when badly overdue (raw model.forward drivers with no generator, e.g. perf.py)
        ip.moe_cpu_swap_pending = True
        # Servers must not reshuffle experts during a long response: the synchronous
        # checkpoint reads create a multi-second latency spike and moving an expert
        # between CPU/GPU kernels can perturb the continuation. Generator-backed
        # callers already run pending sweeps from on_queue_drained(), so allow them to
        # disable the raw-forward fallback without changing the default behaviour.
        inline = os.environ.get("EXL3_MOE_CPU_SWAP_INLINE", "1") != "0"
        if not inline or self._swap_tick_count < 4 * interval:
            return
        run_pending_swap_sweeps(ip)

    def _split_sweep_layer_reset(self):
        self._swap_tick_count = 0

    def _split_sweep_layer(self, budget):
        """One placement sweep for this layer: repeatedly swap the hottest CPU-resident
        expert with the coldest GPU-resident one while the hit-count ratio clears the
        hysteresis threshold. The promoted expert's tensors are re-read from the checkpoint
        into the demoted expert's slot IN PLACE, so every baked pointer (multilinear tables,
        captured graphs, fused stacks) stays valid; the demoted expert becomes CPU-resident
        by map update alone, since the worker holds every expert. Skips shape-mismatched
        pairs (per-expert quant width differences). Decays the counts afterwards so the
        stats track recent routing."""
        first = self.cpu_split_first
        hyst = float(os.environ.get("EXL3_MOE_CPU_SWAP_HYST", 2.0))
        mp = self._split_map.cpu()
        hist = self._split_hist.cpu()
        # Preserve the exact pre-decay router-space counts for reproducible placement
        # training. This is written only when EXL3_MOE_CPU_SWAP_HIST_OUT is configured.
        self._split_last_hist = hist.tolist()
        self._split_last_total_hits = float(hist.sum())
        self._split_last_gpu_hits = sum(
            float(hist[r]) for r in range(self.num_experts) if int(mp[r]) < first)
        if self._split_hist_prefill is not None:
            self._split_last_hist_prefill = self._split_hist_prefill.cpu().tolist()
            self._split_last_hist_decode = self._split_hist_decode.cpu().tolist()
        if self._split_lru_sims:
            self._split_last_lru = {}
            for sim in self._split_lru_sims:
                hits, misses = (int(v) for v in sim["counters"].cpu().tolist())
                mass = hits + misses
                self._split_last_lru[str(sim["capacity"])] = {
                    "hits": hits,
                    "misses": misses,
                    "hit_rate": hits / mass if mass else 0.0,
                }
        # Absolute mass floor on top of the ratio test: with short accumulation windows the
        # per-expert counts are only a few hits, and a tail expert with 3 vs a head expert
        # with 1 clears any pure ratio -- the sweep then churns on noise forever (each churn
        # a checkpoint read). Requiring several multiples of the uniform expectation makes
        # the settled state a fixed point: post-settling tail experts sit far below the
        # floor, so sweeps become no-ops
        floor = float(os.environ.get("EXL3_MOE_CPU_SWAP_FLOOR", 8.0)) \
            * float(hist.sum()) / self.num_experts
        fixed = self._split_fixed_first if self._split_mirror_slots else 0
        # Mirror mode never evicts the fixed prefix; only the replaceable mirror slots are
        # candidates. The CPU master for every mirrored expert remains in the worker.
        head = [(float(hist[r]), r) for r in range(self.num_experts)
                if fixed <= int(mp[r]) < first]
        tail = [(float(hist[r]), r) for r in range(self.num_experts) if int(mp[r]) >= first]
        head.sort()
        tail.sort(reverse = True)
        nswaps = 0
        for (c_cold, r_cold), (c_hot, r_hot) in zip(head, tail):
            if nswaps >= budget or c_hot < max(hyst * max(c_cold, 1.0), floor):
                break
            if self._split_swap_experts(r_cold, r_hot, mp):
                nswaps += 1
        if nswaps:
            if os.environ.get("EXL3_MOE_CPU_SWAP_VERIFY"):
                if self._split_mirror_slots:
                    cached = sorted(int(v) for v in mp.tolist() if fixed <= int(v) < first)
                    assert cached == list(range(fixed, first)), \
                        f"{self.key}: mirror slots are not occupied exactly once after sweep"
                    for r, v in enumerate(mp.tolist()):
                        if v >= first:
                            assert v == int(self._split_cpu_physical[r]), \
                                f"{self.key}: CPU master mapping changed for router expert {r}"
                else:
                    assert mp.sort().values.equal(torch.arange(self.num_experts)), \
                        f"{self.key}: placement map is not a permutation after sweep"
            self._split_map.copy_(mp.to(self._split_map.device))
        # The default half-life keeps dynamic placement responsive. Offline routing
        # studies can set the factor to 1.0 to export exact cumulative counts across
        # a multi-request agent trajectory without changing placement behavior.
        hist_decay = float(os.environ.get("EXL3_MOE_CPU_SWAP_HIST_DECAY", "0.5"))
        if not 0.0 <= hist_decay <= 1.0:
            raise ValueError("EXL3_MOE_CPU_SWAP_HIST_DECAY must be in [0, 1]")
        self._split_hist.mul_(hist_decay)
        if self._split_hist_prefill is not None:
            self._split_hist_prefill.mul_(hist_decay)
            self._split_hist_decode.mul_(hist_decay)
        return nswaps

    def _split_swap_experts(self, r_cold, r_hot, mp):
        """Promote router expert r_hot into the GPU slot of r_cold (checkpoint read +
        in-place tensor copy), demote r_cold by map update. mp is the host-side map, updated
        on success."""
        stc = self.config.stc
        slot = int(mp[r_cold])
        full_g, full_u, full_d = self._split_saved_lists_ref()
        proj = ([(full_g, self.gates)] if self.gated else []) \
            + [(full_u, self.ups), (full_d, self.downs)]
        pairs = []
        for full, cur in proj:
            src_key = full[r_hot].key
            dst = cur[slot].inner
            shp = stc.list_tensors(src_key)[src_key + ".trellis"]["shape"]
            if list(shp) != list(dst.trellis.shape):
                return False
            pairs.append((src_key, dst))
        # The demoted expert also must match the worker slot's tenant (= the promoted
        # expert's shapes, since they exchange homes) for the in-place arena copy
        for full, _ in proj:
            k_cold = full[r_cold].key
            k_hot = full[r_hot].key
            sc = stc.list_tensors(k_cold)[k_cold + ".trellis"]["shape"]
            sh = stc.list_tensors(k_hot)[k_hot + ".trellis"]["shape"]
            if list(sc) != list(sh):
                return False

        for src_key, dst in pairs:
            dst.trellis.copy_(stc.get_tensor(src_key + ".trellis", dst.trellis.device))
            dst.suh.copy_(stc.get_tensor(src_key + ".suh", dst.suh.device, float2half = True))
            dst.svh.copy_(stc.get_tensor(src_key + ".svh", dst.svh.device, float2half = True))
            if getattr(dst, "bias", None) is not None:
                b = stc.get_tensor(src_key + ".bias", dst.bias.device, optional = True,
                                   float2half = True)
                if b is not None:
                    dst.bias.copy_(b)

        if not self._split_mirror_slots:
            # Ordinary exchange mode: demote r_cold into the worker slot r_hot vacated.
            # The child re-reads the weights from its checkpoint handle into the arena.
            cpu_local = int(mp[r_hot]) - self.cpu_split_first
            keys = ([full_g[r_cold].key] if self.gated else []) \
                + [full_u[r_cold].key, full_d[r_cold].key]
            self.cpu_host.install_expert(self.cpu_layer_idx, cpu_local, keys)
            aux = self.cpu_host.aux.get(self.cpu_layer_idx)
            if aux:
                names = ([("suh_g", full_g), ("svh_g", full_g), ("bias_g", full_g)] if self.gated else []) \
                    + [("suh_u", full_u), ("svh_u", full_u), ("bias_u", full_u),
                       ("suh_d", full_d), ("svh_d", full_d), ("bias_d", full_d)]
                for name, full in names:
                    lst = aux.get(name)
                    if lst is None or lst[cpu_local] is None:
                        continue
                    suffix = "." + name.split("_")[0]
                    t = stc.get_tensor(full[r_cold].key + suffix, lst[cpu_local].device,
                                       optional = name.startswith("bias"), float2half = True)
                    if t is not None:
                        lst[cpu_local].copy_(t)

        if os.environ.get("EXL3_MOE_CPU_SWAP_VERIFY"):
            # Read back the promoted slot and compare against the checkpoint (catches any
            # copy/stream/layout unfaithfulness in the real sweep context)
            torch.cuda.synchronize()
            for src_key, dst in pairs:
                for tname, live in (("trellis", dst.trellis), ("suh", dst.suh), ("svh", dst.svh)):
                    fresh = stc.get_tensor(src_key + "." + tname, live.device,
                                           float2half = tname != "trellis")
                    assert torch.equal(fresh, live), \
                        f"swap verify failed: {src_key}.{tname}"
        if self._split_mirror_slots:
            # The CPU master never moved: evicting the old mirror tenant restores its
            # immutable CPU physical id, while the promoted router id acquires the slot.
            mp[r_cold] = int(self._split_cpu_physical[r_cold])
            mp[r_hot] = slot
        else:
            mp[r_cold], mp[r_hot] = int(mp[r_hot]), slot
        return True

    def _split_saved_lists_ref(self):
        g, u, d = self._split_saved[0], self._split_saved[1], self._split_saved[2]
        return g, u, d
