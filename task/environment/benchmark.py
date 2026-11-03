"""Transfer-only Qwen3.5 state benchmark. Evaluator authority; do not edit during research."""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import statistics
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
LENGTHS = [4096, 8192, 32768, 65536, 4097, 10034, 32749, 65521]
BLOCK = 128
PROTOCOL_VERSION = "0.2.0"
LAYOUTS = ("contiguous", "fragmented", "fragmented_runs")


def sync():
    for i in range(4):
        torch.cuda.synchronize(i)


def reference(plan):
    """Naive Torch: gather, peer-copy, scatter each overlapping rank/head range.
    Q/K/V convolution segments must be sliced separately for heterogeneous TP.
    """
    src, dst = plan["src"], plan["dst"]
    p, d = plan["p"], plan["d"]
    sm, dm, ss, ds = [
        plan[k] for k in ["src_pages", "dst_pages", "src_slots", "dst_slots"]
    ]
    for dr in range(d):
        target = dst[dr]
        dg = p + dr
        di = dm.to(f"cuda:{dg}")
        dsi = ds.to(f"cuda:{dg}")
        for sr in range(p):
            origin = src[sr]
            si = sm.to(f"cuda:{sr}")
            ssi = ss.to(f"cuda:{sr}")
            # Full attention: global KV head overlap, 2 heads total.
            lo = max(sr * 2 // p, dr * 2 // d)
            hi = min((sr + 1) * 2 // p, (dr + 1) * 2 // d)
            if lo < hi:
                sl = slice(lo - sr * 2 // p, hi - sr * 2 // p)
                dl = slice(lo - dr * 2 // d, hi - dr * 2 // d)
                for layer in range(10):
                    packed = origin["kv"][layer, :, sl].index_select(0, si)
                    remote = packed.to(f"cuda:{dg}")
                    target["kv"][layer, :, dl].index_copy_(0, di, remote)
            # Fixed recurrent state: 32 global value heads.
            lo = max(sr * 32 // p, dr * 32 // d)
            hi = min((sr + 1) * 32 // p, (dr + 1) * 32 // d)
            if lo < hi:
                sl = slice(lo - sr * 32 // p, hi - sr * 32 // p)
                dl = slice(lo - dr * 32 // d, hi - dr * 32 // d)
                packed = origin["recurrent"][:, :, sl].index_select(1, ssi)
                target["recurrent"][:, :, dl].index_copy_(
                    1, dsi, packed.to(f"cuda:{dg}")
                )
            # Three independently TP-sharded channel segments, not one contiguous shard.
            for seg, total in enumerate([2048, 2048, 4096]):
                so = sum([2048, 2048, 4096][:seg]) // p
                do = sum([2048, 2048, 4096][:seg]) // d
                lo = max(sr * total // p, dr * total // d)
                hi = min((sr + 1) * total // p, (dr + 1) * total // d)
                if lo >= hi:
                    continue
                sl = slice(so + lo - sr * total // p, so + hi - sr * total // p)
                dl = slice(do + lo - dr * total // d, do + hi - dr * total // d)
                packed = origin["conv"][:, :, :, sl].index_select(1, ssi)
                target["conv"][:, :, :, dl].index_copy_(1, dsi, packed.to(f"cuda:{dg}"))


def derive_seed(seed, tag):
    return int.from_bytes(
        hashlib.blake2b(f"{seed}/{tag}".encode(), digest_size=8).digest(), "little"
    ) & ((1 << 63) - 1)


def random_pages(cap, n, generator, run_length=None):
    if run_length is None:
        return torch.randperm(cap, generator=generator)[:n]
    # Shuffle bounded physical runs; order within each run remains local before
    # logical-pair scheduling, giving a distinct clustered-fragmentation regime.
    chunks = []
    for run in torch.randperm(
        math.ceil(cap / run_length), generator=generator
    ).tolist():
        chunks.extend(range(run * run_length, min(cap, (run + 1) * run_length)))
    return torch.tensor(chunks[:n], dtype=torch.int64)


def maps(bs, ctx, layout, seed):
    if isinstance(layout, bool):
        layout = "fragmented" if layout else "contiguous"
    if layout not in LAYOUTS:
        raise ValueError("Unknown physical layout")
    n = bs * math.ceil(ctx / BLOCK)
    cap = math.ceil(n * 1.5) + 17
    slots = 2 * bs + 7

    def gen(tag):
        return torch.Generator().manual_seed(derive_seed(seed, tag))

    if layout == "contiguous":
        a = torch.arange(n) + 3
        b = torch.arange(n) + 9
    else:
        a = random_pages(
            cap, n, gen("source-pages"), 8 if layout == "fragmented_runs" else None
        )
        b = random_pages(
            cap, n, gen("destination-pages"), 5 if layout == "fragmented_runs" else None
        )
        # Flattened logical identity is request*pages_per_request+token_page.
        # Shuffle the pair schedule jointly, interleaving requests without changing
        # which source logical page belongs at which destination logical page.
        schedule = torch.randperm(n, generator=gen("pair-schedule"))
        a = a[schedule]
        b = b[schedule]
    return dict(
        src_pages=a,
        dst_pages=b,
        src_slots=torch.randperm(slots, generator=gen("source-state-slots"))[:bs],
        dst_slots=torch.randperm(slots, generator=gen("destination-state-slots"))[:bs],
        cap=cap,
        slotcap=slots,
    )


def allocate(p, d, bs, ctx, layout, seed):
    plan = maps(bs, ctx, layout, seed)
    plan.update(p=p, d=d, bs=bs, ctx=ctx, src=[], dst=[])
    for group, tp, offset in [("src", p, 0), ("dst", d, p)]:
        for rank in range(tp):
            dev = f"cuda:{offset + rank}"
            plan[group].append(
                {
                    "kv": torch.empty(
                        (10, plan["cap"], 2 // tp, BLOCK, 512),
                        dtype=torch.bfloat16,
                        device=dev,
                    ),
                    "conv": torch.empty(
                        (30, plan["slotcap"], 3, 8192 // tp),
                        dtype=torch.bfloat16,
                        device=dev,
                    ),
                    "recurrent": torch.empty(
                        (30, plan["slotcap"], 32 // tp, 128, 128),
                        dtype=torch.float32,
                        device=dev,
                    ),
                }
            )
    return plan


def random_finite_payload(tensor):
    # Broad bit patterns prevent the workload becoming a small-integer copy toy.
    # Source caches are opaque bytes; no model arithmetic is performed here.
    if tensor.dtype == torch.bfloat16:
        bits = tensor.view(torch.int16)
        lo, hi, mask = -(1 << 15), 1 << 15, 0x7F80
    elif tensor.dtype == torch.float32:
        bits = tensor.view(torch.int32)
        lo, hi, mask = -(1 << 31), 1 << 31, 0x7F800000
    else:
        raise TypeError("Unsupported state dtype")
    bits.random_(lo, hi)
    bits.masked_fill_(
        (bits & mask) == mask, 0
    )  # Exclude NaN/Inf, preserve signed zero.


def bit_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    dtype = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return torch.equal(a.view(dtype), b.view(dtype))


def fill(plan, generation):
    for r, src in enumerate(plan["src"]):
        torch.cuda.set_device(r)
        for t in src.values():
            random_finite_payload(t)
    poison = -777.0 if generation % 2 == 0 else 333.0
    for target in plan["dst"]:
        for t in target.values():
            t.fill_(poison)
    return poison


def check(plan, snapshot, poison):
    """Independent checker maps global heads and each convolution segment.
    Compare selected pages/slots and require all holes to preserve poison.
    """
    p, d = plan["p"], plan["d"]
    sm = plan["src_pages"]
    dm = plan["dst_pages"]
    ss = plan["src_slots"]
    ds = plan["dst_slots"]
    for dr, target in enumerate(plan["dst"]):
        dev = target["kv"].device
        di = dm.to(dev)
        dsi = ds.to(dev)
        for name, count in [("kv", 2), ("recurrent", 32)]:
            axis = 2
            for local in range(count // d):
                glob = dr * (count // d) + local
                sr = glob // (count // p)
                sh = glob % (count // p)
                source = snapshot[sr][name]
                idx = sm if name == "kv" else ss
                exp = source[:, :, sh].index_select(1, idx.to(source.device)).to(dev)
                actual = target[name][:, :, local].index_select(
                    1, di if name == "kv" else dsi
                )
                if not bit_equal(actual, exp):
                    raise AssertionError(f"{name} mismatch dr={dr}, head={glob}")
        for seg, total in enumerate([2048, 2048, 4096]):
            for sr in range(p):
                lo = max(sr * total // p, dr * total // d)
                hi = min((sr + 1) * total // p, (dr + 1) * total // d)
                if lo >= hi:
                    continue
                prefix = sum([2048, 2048, 4096][:seg])
                slo = prefix // p + lo - sr * total // p
                dlo = prefix // d + lo - dr * total // d
                source = snapshot[sr]["conv"]
                expected = (
                    source[:, :, :, slo : slo + hi - lo]
                    .index_select(1, ss.to(source.device))
                    .to(dev)
                )
                actual = target["conv"][:, :, :, dlo : dlo + hi - lo].index_select(
                    1, dsi
                )
                if not bit_equal(actual, expected):
                    raise AssertionError(f"conv mismatch segment={seg},dr={dr},sr={sr}")
        for name, t in target.items():
            used = dm if name == "kv" else ds
            mask = torch.ones(t.shape[1], dtype=torch.bool)
            mask[used] = False
            for ids in mask.nonzero().flatten().split(64):
                if not bool((t.index_select(1, ids.to(dev)) == poison).all()):
                    raise AssertionError(f"overwritten hole {name}")
    # Selected final-page tails are allowed to be copied; each page here includes payload
    # in its tail, enabling byte-exact full-page validation without forwarding the model.


def snapshot_source(plan):
    return [{name: t.clone() for name, t in r.items()} for r in plan["src"]]


def check_source(plan, snapshot):
    for original, expected in zip(plan["src"], snapshot):
        for name, t in original.items():
            if not bit_equal(t, expected[name]):
                raise AssertionError("Source changed: " + name)


def candidate_view(plan):
    """Candidate receives private metadata copies and tensor aliases, never Judge state.
    Tensor detach() gives independent shape/stride metadata while sharing payload storage.
    """
    view = {
        key: value
        for key, value in plan.items()
        if key not in ("src", "dst", "src_pages", "dst_pages", "src_slots", "dst_slots")
    }
    for key in ("src_pages", "dst_pages", "src_slots", "dst_slots"):
        view[key] = plan[key].clone()
    for group in ("src", "dst"):
        view[group] = [
            {name: t.detach() for name, t in rank.items()} for rank in plan[group]
        ]
    return view


def validate_candidate_view(view, trusted):
    """Reject map/scalar/tensor-descriptor mutation, using only evaluator-owned state."""
    if type(view) is not dict or view.keys() != trusted.keys():
        raise AssertionError("Plan structure changed")
    for key in ("p", "d", "bs", "ctx", "cap", "slotcap"):
        if type(view[key]) is not int or view[key] != trusted[key]:
            raise AssertionError("Plan scalar changed: " + key)
    for key in ("src_pages", "dst_pages", "src_slots", "dst_slots"):
        given, expected = view[key], trusted[key]
        if (
            type(given) is not torch.Tensor
            or given.device != expected.device
            or given.dtype != expected.dtype
            or given.shape != expected.shape
            or not torch.equal(given, expected)
        ):
            raise AssertionError("Plan map changed: " + key)
    for group in ("src", "dst"):
        if type(view[group]) is not list or len(view[group]) != len(trusted[group]):
            raise AssertionError("Plan ranks changed")
        for given_rank, expected_rank in zip(view[group], trusted[group]):
            if (
                type(given_rank) is not dict
                or given_rank.keys() != expected_rank.keys()
            ):
                raise AssertionError("Plan state keys changed")
            for name, expected in expected_rank.items():
                given = given_rank[name]
                if (
                    type(given) is not torch.Tensor
                    or given.dtype != expected.dtype
                    or given.device != expected.device
                    or given.shape != expected.shape
                    or given.stride() != expected.stride()
                    or given.storage_offset() != expected.storage_offset()
                    or given.data_ptr() != expected.data_ptr()
                ):
                    raise AssertionError("Plan tensor descriptor changed: " + name)


def validate_candidate_source(source):
    """Reject direct checker/timer/file/process tampering before candidate import.

    This is a bounded source policy, not a proof against obfuscated native code.
    Native-code containment remains an official Harness validation requirement.
    """
    tree = ast.parse(source)
    # The exact baseline adapter is an explicitly allowed baseline submission.
    if source.strip() == "from benchmark import reference as transfer":
        return
    allowed_roots = {"torch", "triton", "math"}
    host_modules = {
        "os",
        "sys",
        "time",
        "inspect",
        "ctypes",
        "subprocess",
        "builtins",
        "importlib",
        "gc",
        "modules",
        "environ",
    }
    forbidden_calls = {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "input",
    }

    def root_name(node):
        while isinstance(node, ast.Attribute):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed_roots:
                    raise ValueError(
                        "Candidate import outside transfer API: " + alias.name
                    )
        elif isinstance(node, ast.ImportFrom):
            if (
                node.level
                or not node.module
                or node.module.split(".")[0] not in allowed_roots
            ):
                raise ValueError("Candidate import outside transfer API")
            if node.module == "torch.utils.cpp_extension" and any(
                alias.name not in {"load", "load_inline"} for alias in node.names
            ):
                raise ValueError(
                    "Only trusted extension build entrypoints may be imported"
                )
            if any(
                alias.name in host_modules or alias.name.startswith("__")
                for alias in node.names
            ):
                raise ValueError("Host module alias import prohibited")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Candidate builtin/private introspection prohibited")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in forbidden_calls
        ):
            raise ValueError(
                "Candidate reflection/I-O/evaluation call prohibited: " + node.func.id
            )
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in host_modules:
                raise ValueError("Candidate private/module introspection prohibited")
            if (
                isinstance(node.ctx, (ast.Store, ast.Del))
                and root_name(node) in allowed_roots
            ):
                raise ValueError("Candidate module monkeypatch prohibited")
    host_access = r"\b(?:system|popen|fopen|freopen|dlopen|dlsym|ptrace|fork|execve|syscall|PyImport_ImportModule|PyRun_SimpleString)\s*\(|std::(?:filesystem|ifstream|ofstream|fstream)"
    if re.search(host_access, source):
        raise ValueError(
            "Candidate native host process/file/introspection API prohibited"
        )


def load_candidate(path):
    validate_candidate_source(Path(path).read_text())
    spec = importlib.util.spec_from_file_location("candidate", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.transfer


def run_case(fn, p, d, bs, ctx, layout, warmup, samples, seed):
    plan = allocate(p, d, bs, ctx, layout, seed)
    try:
        for generation in [1, 2]:
            poison = fill(plan, generation)
            sync()
            before = snapshot_source(plan)
            exposed = candidate_view(plan)
            fn(exposed)
            sync()
            validate_candidate_view(exposed, plan)
            check(plan, before, poison)
            check_source(plan, before)
            del before
            # Next generation permutes maps again, including slot reuse.
            plan.update(maps(bs, ctx, layout, seed + generation + 1337))
        times = []
        for i in range(warmup + samples):
            plan.update(maps(bs, ctx, layout, seed + i + 8192))
            poison = fill(plan, i + 3)
            before = snapshot_source(plan)
            sync()
            exposed = candidate_view(plan)
            start = time.perf_counter_ns()
            fn(exposed)
            sync()
            elapsed = (time.perf_counter_ns() - start) / 1000
            if i >= warmup:
                times.append(elapsed)
            # Check the measured result too; every measured invocation has new data/maps.
            validate_candidate_view(exposed, plan)
            check(plan, before, poison)
            check_source(plan, before)
            del before, exposed
        allocated = [torch.cuda.max_memory_allocated(i) for i in range(4)]
        physical = (
            2 * bs * math.ceil(ctx / BLOCK) * BLOCK * 2 * 256 * 2 * 10 + bs * 64471040
        )
        return dict(
            p=p,
            d=d,
            bs=bs,
            ctx=ctx,
            layout=layout,
            correct=True,
            median_us=statistics.median(times),
            samples_us=times,
            useful_bytes=bs * (ctx * 20480 + 64471040),
            page_payload_bytes=physical,
            peak_allocated=allocated,
        )
    finally:
        del plan
        sync()
        torch.cuda.empty_cache()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--candidate")
    a.add_argument("--output", required=True)
    a.add_argument("--smoke", action="store_true")
    a.add_argument("--warmup", type=int, default=3)
    a.add_argument("--samples", type=int, default=10)
    a.add_argument(
        "--seed",
        type=int,
        help="Optional reproducible development map seed; default uses OS entropy",
    )
    args = a.parse_args()
    base_seed = args.seed if args.seed is not None else secrets.randbits(63)
    if torch.cuda.device_count() != 4:
        raise RuntimeError("Require exactly four delegated GPUs")
    for i in range(4):
        for j in range(4):
            if i != j and not torch.cuda.can_device_access_peer(i, j):
                raise RuntimeError(f"No peer access {i}->{j}")
    fn = reference if not args.candidate else load_candidate(args.candidate)
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": [
            {
                "name": torch.cuda.get_device_properties(i).name,
                "capability": list(torch.cuda.get_device_capability(i)),
                "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(4)
        ],
        "protocol_version": PROTOCOL_VERSION,
        "seed_mode": "provided" if args.seed is not None else "os-random",
        "mode": "torch-reference" if not args.candidate else Path(args.candidate).name,
        "scope": "synthetic transfer kernels only; no model forward; no vLLM IPC or NIXL service overhead",
        "block_size": BLOCK,
        "started_at": time.time(),
    }
    (output.with_suffix(".environment.json")).write_text(
        json.dumps(env, indent=2, default=str)
    )
    with output.open("w") as out:
        for p, d in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            for ctx in [4097] if args.smoke else LENGTHS:
                for bs in [1] if args.smoke else [1, 4]:
                    for layout in LAYOUTS:
                        for i in range(4):
                            torch.cuda.reset_peak_memory_stats(i)
                        row = run_case(
                            fn,
                            p,
                            d,
                            bs,
                            ctx,
                            layout,
                            args.warmup,
                            args.samples,
                            seed=derive_seed(base_seed, f"{p}/{d}/{bs}/{ctx}/{layout}"),
                        )
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        print(json.dumps(row), flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
