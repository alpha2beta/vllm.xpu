"""G3 parity: §11.5 token-parallel grouped verify path (1 < T <= 4).

`Exl3MoEMethod._grouped_mlp_verify` replaces the P4-03 prefill path for small
multi-token batches (MTP verify, and any 2-4 row batch). It is *not* spec-only:
`apply()` selects it whenever `1 < tokens <= 4`, so it also sits on the
certified Gate 4 path. This battery certifies it:

* V-1 kernel parity — bit-exact against the per-row P4-01 `_expert_mlp_cached`
  reference, reproducing §11.5's algebra (fp32, token-major slot order) for
  T ∈ {2, 3, 4} × {bf16, fp16} and for disjoint / shared / router routing.
* V-2 apply() parity — `apply()` through the verify branch vs `apply()` with
  every grouped path latched off (P2-01 eager loop).
* V-3 dispatch ladder — `apply()` calls the verify path for T ∈ {2,3,4}, not
  for T = 1 (P4-02 decode) and not for T = 5 (P4-03 prefill).
* V-4 latch isolation — a forced failure latches `_verify_supported` only,
  warns exactly once, leaves P4-02 decode working, and `apply()` stays correct
  through the production ladder (§11.6).
* V-5 P4-03 prefill parity — **pre-existing failure on this checkpoint**
  (K=4 / cb=2), reported non-fatally unless ``--strict``.

Findings recorded 2026-10-27 (evidence: `logs/moe-p3/server-g0-trace.log`,
`logs/moe-p3/g3-grouped-verify.json`):

1. The verify path is bit-exact (V-1, 18/18) and `apply()` routes to it for
   exactly T ∈ {2,3,4} (V-3).
2. P4-03 is **inert in production**: its SYCL JIT fails in the serving
   process (`RuntimeError: The program was built for 1 devices`, root cause
   `LLVM ERROR: VISA builder API call failed ... Compiler error emitted in
   code generator`), which latches `_grouped_gemm_supported` off on the first
   prefill — so every prefill uses the eager loop. Run the launcher with
   `EXL3_TRACE_PATHS=1` to reproduce.
3. When P4-03 *does* build (in-process), it computes **wrong values for
   K=4/cb=2** — the kernel was only ever certified on K=3/cb=1. V-5 measures
   it. Fixing the JIT error alone would activate a numerically broken kernel.

    .venv/bin/python tests/test_grouped_verify.py [--model-dir DIR]
                        [--json-out F] [--strict]
"""

from __future__ import annotations

import argparse
import json
import traceback
import warnings
from pathlib import Path

import torch

import exl3xpu.moe as moe_mod
from _moe_harness import DEFAULT_MODEL, HIDDEN, NUM_EXPERTS, Harness

TS = (2, 3, 4)
TOPK = 8
EAGER_MAX_ABS = 0.02   # bf16/fp16 bound vs the P2-01 loop (fp32 sum order differs)
PREFILL_MAX_ABS = 0.02 # P4-03 absolute bound (companion to the relative one)
# Correct paths agree with the reference to <=0.2 % of the output scale; a
# scaling-independent bound is required because random-input outputs are O(0.03)
# and an absolute tolerance alone hides a 23 % relative error.
MAX_REL = 0.05
LATCH_ATTRS = ("_verify_supported", "_verify_latch_logged",
               "_grouped_supported", "_grouped_gemm_supported")

RESULT: list[dict] = []
PRE_EXISTING: list[dict] = []


def _reset_latches(method) -> None:
    """Back to first-call state so every case probes the paths afresh."""
    for attr in LATCH_ATTRS:
        if hasattr(method, attr):
            delattr(method, attr)


def _routing(x: torch.Tensor, kind: str, h: Harness):
    """top-k ids/weights in [T, 8]; non-uniform weights in every case."""
    T = x.shape[0]
    if kind == "router":
        _, ids, wts = h.router(x)
        return ids.view(T, -1), wts.view(T, -1)
    if kind == "shared":            # every token picks the same experts
        base = [3, 7, 42, 100, 255, 0, 17, 200]
        ids = torch.tensor([base for _ in range(T)], dtype=torch.int32, device=x.device)
    elif kind == "disjoint":        # no overlap at all (measured Sage behaviour)
        ids = torch.stack([
            torch.tensor([(t * TOPK + j) % NUM_EXPERTS for j in range(TOPK)],
                         dtype=torch.int32, device=x.device)
            for t in range(T)
        ])
    else:
        raise ValueError(kind)
    # Distinct, non-uniform weights: catches token/expert flattening mistakes.
    wts = torch.stack([
        torch.tensor([1.0 + 0.37 * t + 0.11 * j for j in range(TOPK)],
                     dtype=torch.float32, device=x.device)
        for t in range(T)
    ])
    return ids, wts


def _reference_verify(h: Harness, x: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor):
    """§11.5 reference: per-row P4-01 MLPs, fp32, token-major k-slot sum."""
    T, k = ids.shape
    per = torch.zeros(T, k, HIDDEN, dtype=torch.float32, device=x.device)
    for t in range(T):
        for j in range(k):
            y = h.method._expert_mlp_cached(int(ids[t, j]), x[t : t + 1])
            per[t, j] = y.reshape(-1).float() * float(wts[t, j])
    return per.view(T, k, -1).sum(dim=1).to(x.dtype)


def _apply_mode(h: Harness, x, ids, wts, mode: str):
    """apply() under a specific path configuration.

    verify — every grouped path enabled (T<=4 takes §11.5)
    eager  — all grouped paths latched off: the P2-01 loop (production's
             steady state for prefill, since P4-03 latches off there)
    """
    _reset_latches(h.method)
    if mode == "eager":
        h.method._verify_supported = False
        h.method._grouped_supported = False
        h.method._grouped_gemm_supported = False
    elif mode != "verify":
        raise ValueError(mode)
    try:
        return h.method.apply(h.layer, x, wts, ids, None, None)
    finally:
        _reset_latches(h.method)


def _record(case: str, ok: bool, detail: dict, fatal: bool = True) -> None:
    bucket = RESULT if fatal else PRE_EXISTING
    bucket.append({"case": case, "ok": ok, **detail})
    tag = "OK  " if ok else ("FAIL" if fatal else "KNOWN-FAIL")
    print(f"{tag} {case}: " + ", ".join(f"{k}={v}" for k, v in detail.items()))


# --------------------------------------------------------------------------- #
# V-1 / V-2 — parity
# --------------------------------------------------------------------------- #
def test_verify_parity(h: Harness, dtype: torch.dtype) -> None:
    dev = torch.device("xpu")
    for T in TS:
        for kind in ("router", "disjoint", "shared"):
            torch.manual_seed(7000 + T)
            x = torch.randn(T, HIDDEN, dtype=dtype, device=dev)
            ids, wts = _routing(x, kind, h)

            _reset_latches(h.method)
            got = h.method._grouped_mlp_verify(h.layer, x, ids, wts)
            assert got is not None, f"verify returned None (T={T}, {kind})"
            assert got.shape == (T, HIDDEN) and got.dtype == dtype, (
                f"{tuple(got.shape)}/{got.dtype} != ({T},{HIDDEN})/{dtype}")

            want = _reference_verify(h, x, ids, wts)
            assert want.dtype == dtype
            diff = (got.float() - want.float()).abs().max().item()
            assert torch.equal(got, want), (
                f"V-1 T={T} {kind} [{dtype}]: max_abs={diff:.3e} "
                f"(token/expert mapping or weight order wrong)")
            _record(f"V-1 kernel parity T={T} {kind} {dtype}", True, {"max_abs": diff})

            # V-2: apply() through the verify branch vs the P2-01 eager loop
            out_v = _apply_mode(h, x, ids, wts, "verify")
            out_e = _apply_mode(h, x, ids, wts, "eager")
            assert out_v.dtype == out_e.dtype == dtype
            d2 = (out_v.float() - out_e.float()).abs().max().item()
            scale = max(out_e.float().abs().max().item(), 1e-6)
            ok = d2 <= EAGER_MAX_ABS and d2 / scale <= MAX_REL
            _record(f"V-2 apply vs eager T={T} {kind} {dtype}", ok,
                    {"max_abs": d2, "rel": d2 / scale})
            assert ok, (f"V-2 T={T} {kind} [{dtype}]: verify diverged from the "
                        f"eager loop max_abs={d2:.3e} rel={d2 / scale:.3e} "
                        f"(tol {EAGER_MAX_ABS} / {MAX_REL})")


# --------------------------------------------------------------------------- #
# V-5 — P4-03 prefill (pre-existing, non-fatal unless --strict)
# --------------------------------------------------------------------------- #
def test_prefill_parity(h: Harness, strict: bool) -> None:
    dev = torch.device("xpu")
    for T in (4, 16, 64):
        for kind in ("router", "disjoint"):
            torch.manual_seed(7100 + T)
            x = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=dev)
            ids, wts = _routing(x, kind, h)
            ref = _reference_verify(h, x, ids, wts)
            _reset_latches(h.method)
            h.method._verify_supported = False     # force P4-03 in apply()
            try:
                pf = h.method.apply(h.layer, x, wts, ids, None, None)
                engaged = getattr(h.method, "_grouped_gemm_supported", None) is not False
            finally:
                _reset_latches(h.method)
            assert pf is not None
            if not engaged:                        # P4-03 threw and latched
                _record(f"V-5 prefill T={T} {kind} (P4-03 latched, not exercised)",
                        True, {"fatal": strict}, fatal=strict)
                continue
            d = (pf.float() - ref.float()).abs().max().item()
            scale = max(ref.float().abs().max().item(), 1e-6)
            ok = d <= PREFILL_MAX_ABS and d / scale <= MAX_REL
            _record(f"V-5 prefill vs reference T={T} {kind}", ok,
                    {"max_abs": d, "rel": d / scale,
                     "tol": f"{PREFILL_MAX_ABS}/{MAX_REL}"}, fatal=strict)
            if not ok:
                print(f"      ^ pre-existing: exl3_moe_grouped_gemm is wrong for "
                      f"K={h.method.K}/cb={h.method.cb} (never certified outside "
                      f"K=3/cb=1); inert in production because its SYCL JIT fails "
                      f"and latches the path off.")


# --------------------------------------------------------------------------- #
# V-3 — dispatch ladder
# --------------------------------------------------------------------------- #
def test_dispatch_ladder(h: Harness) -> None:
    dev = torch.device("xpu")
    calls: list[int] = []
    orig = type(h.method)._grouped_mlp_verify

    def spy(self_, layer, x2, ids, wts):
        calls.append(x2.shape[0])
        return orig(self_, layer, x2, ids, wts)

    h.method._grouped_mlp_verify = spy.__get__(h.method)
    try:
        for T in (1, 2, 3, 4, 5):
            torch.manual_seed(8000 + T)
            x = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=dev)
            ids, wts = _routing(x, "router", h)
            calls.clear()
            out = h.method.apply(h.layer, x, wts, ids, None, None)
            assert out.shape == (T, HIDDEN)
            expect = T if T in TS else None
            ok = (calls == [T]) if expect else (not calls)
            _record(f"V-3 dispatch T={T}", ok,
                    {"verify_calls": list(calls), "expected": expect})
            assert ok, f"dispatch ladder T={T}: verify calls {calls}"
    finally:
        del h.method._grouped_mlp_verify
        _reset_latches(h.method)


# --------------------------------------------------------------------------- #
# V-4 — latch isolation
# --------------------------------------------------------------------------- #
def test_latch_isolation(h: Harness) -> None:
    dev = torch.device("xpu")
    _reset_latches(h.method)
    x = torch.randn(2, HIDDEN, dtype=torch.bfloat16, device=dev)
    ids, wts = _routing(x, "shared", h)

    # fp32 activation: the grouped op TORCH_CHECKs ("expected fp16/bf16 x").
    bad = x.float()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert h.method._grouped_mlp_verify(h.layer, bad, ids, wts) is None
        assert h.method._verify_supported is False, "verify did not latch"
        # Second failure must not warn again.
        assert h.method._grouped_mlp_verify(h.layer, bad, ids, wts) is None

    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    _record("V-4a latch + single warning", len(runtime) == 1,
            {"warnings": len(runtime)})
    assert len(runtime) == 1, f"expected exactly 1 warning, got {len(runtime)}"
    assert "grouped verify path disabled" in str(runtime[0].message)

    # P4-02 decode must be untouched by the verify failure — and still exact:
    # apply() at T=1 is the certified Gate 4 path, unchanged by this work.
    assert getattr(h.method, "_grouped_supported", None) is not False, (
        "verify failure leaked into _grouped_supported (certified path!)")
    y1 = h.method.apply(h.layer, x[:1], wts[:1], ids[:1], None, None)
    assert h.method._grouped_mlp_verify(h.layer, x, ids, wts) is None
    assert y1 is not None and torch.isfinite(y1).all()
    decode_used = getattr(h.method, "_grouped_supported", None) is True
    d_dec = (y1.float() - _reference_verify(h, x[:1], ids[:1], wts[:1]).float()
             ).abs().max().item()
    _record("V-4b decode path unaffected", decode_used and d_dec <= EAGER_MAX_ABS,
            {"_grouped_supported": getattr(h.method, "_grouped_supported", None),
             "decode_max_abs": d_dec})
    assert decode_used, "P4-02 decode was disabled by the verify latch"

    # apply() after the latch, in production's steady state (P4-03 is latched
    # off there by its own SYCL JIT failure — see the module docstring).
    h.method._grouped_gemm_supported = False   # what production ends up with
    out = h.method.apply(h.layer, x, wts, ids, None, None)
    ref = _apply_mode(h, x, ids, wts, "eager")
    d = (out.float() - ref.float()).abs().max().item()
    ok = d <= EAGER_MAX_ABS
    _record("V-4c apply() falls back correctly", ok, {"max_abs": d})
    assert ok, f"latched apply() diverged: {d:.3e}"

    _reset_latches(h.method)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="treat the pre-existing P4-03 failure (V-5) as fatal")
    args = ap.parse_args()

    failures: list[str] = []
    k_cb = None
    try:
        h = Harness(args.model_dir)
        h.method._build_exec_cache(h.layer)
        k_cb = {"K": h.method.K, "cb": h.method.cb}
        for dtype in (torch.bfloat16, torch.float16):
            test_verify_parity(h, dtype)
        test_dispatch_ladder(h)
        test_latch_isolation(h)
        test_prefill_parity(h, args.strict)
    except AssertionError as exc:
        failures.append(str(exc))
        traceback.print_exc()
    except Exception:
        failures.append("unexpected error")
        traceback.print_exc()

    passed = sum(1 for r in RESULT if r["ok"])
    print(f"\n{passed}/{len(RESULT)} grouped-verify checks pass")
    if PRE_EXISTING:
        pe = sum(1 for r in PRE_EXISTING if r["ok"])
        print(f"{pe}/{len(PRE_EXISTING)} pre-existing P4-03 checks "
              f"({'fatal' if args.strict else 'reported only, use --strict to fail'})")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "task": "G3-grouped-verify",
            "pass": not failures and passed == len(RESULT),
            "checks": RESULT,
            "pre_existing": PRE_EXISTING,
            "failures": failures,
            "strict": args.strict,
            "k_cb": k_cb,
            "model_dir": args.model_dir,
            "device": torch.xpu.get_device_properties(0).name,
            "torch": torch.__version__,
            "moe_method_module": moe_mod.__file__,
        }, indent=2) + "\n")
        print(f"wrote {args.json_out}")
    if failures or passed != len(RESULT):
        print("failures:", failures or "assertion failure above")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
