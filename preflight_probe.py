#!/usr/bin/env python3
"""preflight_probe.py -- validate the risky pre-quantization steps on the REAL
model dir without quantizing anything.

  1. tokenizer loads through GPTQModel's own loader (Tokenicer) -- the step that
     failed for the tiny random model because of its pad_token_id=-1;
  2. AutoConfig resolves the architecture;
  3. the meta-device *shell* model builds -- this is what makes it possible to
     quantize a 52 GB checkpoint on a 45 GB-RAM host.

Exit code 0 == all checks passed.
"""
from __future__ import annotations

import sys
import traceback


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "/work/src"

    print("=== [1] GPTQModel tokenizer load (Tokenicer) ===", flush=True)
    try:
        from gptqmodel.utils.hf import load_hf_tokenizer
        tok = load_hf_tokenizer(src)
        print("OK:", type(tok).__name__, "| vocab", getattr(tok, "vocab_size", "?"),
              "| pad", repr(getattr(tok, "pad_token", None)),
              getattr(tok, "pad_token_id", None))
    except Exception:
        traceback.print_exc()
        return 1

    print("\n=== [2] AutoConfig ===", flush=True)
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(src)
        tc = getattr(cfg, "text_config", cfg)
        print("OK:", cfg.architectures, "| model_type", cfg.model_type,
              "| layers", getattr(tc, "num_hidden_layers", "?"))
    except Exception:
        traceback.print_exc()
        return 1

    print("\n=== [3] meta-device shell build ===", flush=True)
    try:
        from transformers import AutoConfig, AutoModelForImageTextToText
        from gptqmodel.utils.hf import build_shell_model
        cfg = AutoConfig.from_pretrained(src)
        m = build_shell_model(AutoModelForImageTextToText, config=cfg,
                              device_map={"": "cpu"}, _fast_init=False)
        params = list(m.parameters())
        n_meta = sum(1 for p in params if p.device.type == "meta")
        print(f"OK: shell built; params={len(params)} meta={n_meta} "
              f"(bytes={sum(p.numel() for p in params)/1e9:.2f} G-params)")
        if n_meta != len(params):
            print(f"NOTE: {len(params)-n_meta} params are not on meta")
    except Exception:
        traceback.print_exc()
        return 1

    print("\nALL PREFLIGHT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
