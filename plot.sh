#!/usr/bin/env bash
set -eux

OUTBASE="outputs"
PYTHON="${PYTHON:-/home/mkurzyns/.venv/bin/python3}"
CONFIGS=(b1s4 b2s4 b4s4 b1s8 b2s8)
FIGDIR="fig"
IMG_TYPE=${IMG_TYPE:-png}

mkdir -p "$FIGDIR"

# Build CLI arrays
TS_ALL=$(for c in "${CONFIGS[@]}"; do printf '"%s/pickles_%s/ts.pkl",' "$OUTBASE" "$c"; done | sed 's/,$//')
CFG_ALL=$(printf '"%s",' "${CONFIGS[@]}" | sed 's/,$//')
TS_SINGLE="[\"${OUTBASE}/pickles_b2s8/ts.pkl\"]"
CFG_SINGLE='["b2s8"]'
GPU_SINGLE="[\"${OUTBASE}/profile_trace_b2s8/gpu.pkl\"]"

end_to_end() {
  $PYTHON -m chopper.plots.end_to_end \
    --ts_files="[${TS_ALL}]" --configs="[${CFG_ALL}]" \
    --filename="${FIGDIR}/end_to_end.${IMG_TYPE}"
}

gemm_time() {
  $PYTHON -m chopper.plots.gemm_time \
    --ts_files="[${TS_ALL}]" --configs="[${CFG_ALL}]" \
    --fops='["f_attn_fa","f_mlp_dp","f_mlp_gp","f_mlp_up","f_q_ip","f_k_ip","f_v_ip","f_attn_op","f_lp"]' \
    --bops='["b_attn_fa","b_mlp_dp","b_mlp_gp","b_mlp_up","b_q_ip","b_k_ip","b_v_ip","b_attn_op","b_lp"]' \
    --filename="${FIGDIR}/gemm_time.${IMG_TYPE}"
}

vec_time() {
  $PYTHON -m chopper.plots.vec_time \
    --ts_files="[${TS_ALL}]" --configs="[${CFG_ALL}]" \
    --fops='["f_attn_n","f_mlp_n","f_attn_ra","f_mlp_ra","f_qkv_re","f_mlp_gs"]' \
    --bops='["b_attn_n","b_mlp_n","b_qkv_re","b_mlp_gs","b_mlp_gu"]' \
    --oops='["opt_step","opt_gc"]' \
    --filename="${FIGDIR}/vec_time.${IMG_TYPE}"
}

comm_violin() {
  $PYTHON -m chopper.plots.comm_violin \
    --ts_files="[${TS_ALL}]" --configs="[${CFG_ALL}]" \
    --filename="${FIGDIR}/comm_violin.${IMG_TYPE}"
}

overlap_correlation() {
  $PYTHON -m chopper.plots.overlap_correlation \
    --ts_files="${TS_SINGLE}" --configs="${CFG_SINGLE}" \
    --operators='["b_attn_n","b_mlp_n","b_mlp_up","b_mlp_gp","b_mlp_dp"]' \
    --filename="${FIGDIR}/overlap_correlation.${IMG_TYPE}"
}

overlap_gpus() {
  $PYTHON -m chopper.plots.overlap_gpus \
    --ts_files="${TS_SINGLE}" --configs="${CFG_SINGLE}" \
    --operator=f_attn_op \
    --filename="${FIGDIR}/overlap_gpus.${IMG_TYPE}"
}

overlap_confs() {
  $PYTHON -m chopper.plots.overlap_confs \
    --ts_files="[${TS_ALL}]" --configs="[${CFG_ALL}]" \
    --operator=f_attn_fa \
    --filename="${FIGDIR}/overlap_confs.${IMG_TYPE}"
}

launch_overhead() {
  $PYTHON -m chopper.plots.launch_overhead \
    --ts_files="[${TS_ALL}]" --configs="[${CFG_ALL}]" \
    --lops='["b_mlp_dp","opt_gc","f_attn_n"]' \
    --rops='["f_ie","b_ie","f_ln"]' \
    --filename="${FIGDIR}/launch_overhead_bars.${IMG_TYPE}"
}

lead_and_throughput() {
  $PYTHON -m chopper.plots.lead_and_throughput \
    --ts_files="[${TS_ALL}]" --configs="[${CFG_ALL}]" \
    --filename="${FIGDIR}/Straggler_Leader.${IMG_TYPE}"
}

total_power() {
  $PYTHON -m chopper.plots.total_power \
    --metric_files="${GPU_SINGLE}" --configs="${CFG_SINGLE}" \
    --filename="${FIGDIR}/total_power.${IMG_TYPE}"
}

average_power_freq() {
  $PYTHON -m chopper.plots.average_power_freq \
    --gpu_files="${GPU_SINGLE}" --configs="${CFG_SINGLE}" \
    --filename="${FIGDIR}/average_power_freq.${IMG_TYPE}"
}

all() {
  end_to_end
  gemm_time
  vec_time
  comm_violin
  overlap_correlation
  overlap_gpus
  overlap_confs
  launch_overhead
  lead_and_throughput
  total_power
  average_power_freq
}

"${1:-all}"
