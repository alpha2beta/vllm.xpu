# EXL3 Arc 140V baseline environment (E1.4). Source before any EXL3 process:
#   source logs/<run>/baseline-env.sh
# Pinned plugin: 15ded2f3add148c4db3c900cba7de878238f53bc + logs/<run>/no-dpas.patch

# Never apply the plugin's vLLM source-string patches on vLLM 0.30.0.
export EXL3_VLLM_PATCHES=0
# Route M>8 to reconstruct+GEMM; vector kernel covers M<=8 after the patch.
export EXL3_SMALL_M_MAX=8
# No INT8/XMX prefill on 140V baseline.
export EXL3_INT8_PREFILL=0
# ESIMD first; Triton only via explicit EXL3_BACKEND=triton runs (E3.4).
export EXL3_BACKEND=auto
# 1024-column FP16 reconstruction slices (alignment: 1024 % 128 == 0).
export EXL3_RECON_SLICE_N=1024
# No oneDNN tree on this machine; build script would skip it anyway.
export EXL3_NO_DNNL=1
# K=4/6 mul1 are default instantiations; no -DEXL3_ALL_CODEBOOKS (audit: 408xK4 + 1xK6).
# unset EXL3_FLAGS
# Fused single-kernel path stays off (default); fused vec covers MR<=2 only.
# Do not set EXL3_DRAFT_VOCAB (no MTP in the text baseline).

# Scratch sizing (fp16, RECON_SLICE_N=1024):
#   worst Kdim = 17408 (MLP up/gate/down) -> 17408*1024*2 B = ~34 MiB per slice
#   lm_head Kdim 5120 -> 5120*1024*2 B = 10 MiB per slice
#   xh buffer [S,M,k] scales with prefill M (decode M=1 is negligible);
#   peak measured at E4.2, not assumed here.
