#!/usr/bin/env python3
"""Debug: print the real rotary class + positions shape at the failing attention."""
import sys

import vllm.model_executor.models.qwen3_next as q3n

_orig_init = q3n.Qwen3NextAttention.__init__
def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    print(f"[DBG] rope class={type(self.rotary_emb).__name__} "
          f"section={getattr(self.rotary_emb, 'mrope_section', None)} "
          f"interleaved={getattr(self.rotary_emb, 'mrope_interleaved', None)}",
          file=sys.stderr, flush=True)
q3n.Qwen3NextAttention.__init__ = _init

_orig_fwd = q3n.Qwen3NextAttention.forward
_n = [0]
def _fwd(self, positions, hs):
    _n[0] += 1
    if _n[0] <= 2:
        print(f"[DBG] forward#{_n[0]} positions.shape={tuple(positions.shape)} "
              f"hidden={tuple(hs.shape)} rope={type(self.rotary_emb).__name__}",
              file=sys.stderr, flush=True)
    return _orig_fwd(self, positions, hs)
q3n.Qwen3NextAttention.forward = _fwd

from vllm.entrypoints.launchers.api_server.entry import main  # noqa: E402

if __name__ == "__main__":
    main()
