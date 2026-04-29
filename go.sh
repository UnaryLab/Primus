#!/usr/bin/env bash
set -eux

CONFIGS=(
  "./chopper_configs/llama3.1_70B-BF16-pretrain-b1s4.yaml"
  "./chopper_configs/llama3.1_70B-BF16-pretrain-b2s4.yaml"
  "./chopper_configs/llama3.1_70B-BF16-pretrain-b4s4.yaml"
  "./chopper_configs/llama3.1_70B-BF16-pretrain-b1s8.yaml"
  "./chopper_configs/llama3.1_70B-BF16-pretrain-b2s8.yaml"
  # "./chopper_configs/llama3.1_8B-BF16-pretrain-b1s4.yaml"
  # "./chopper_configs/llama3.1_8B-BF16-pretrain-b2s4.yaml"
  # "./chopper_configs/llama3.1_8B-BF16-pretrain-b4s4.yaml"
  # "./chopper_configs/llama3.1_8B-BF16-pretrain-b1s8.yaml"
  # "./chopper_configs/llama3.1_8B-BF16-pretrain-b2s8.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
  # Extract e.g. "profile_trace_b2s8" from config
  OUTDIR=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg['modules']['pre_trainer']['overrides']['profiling']['save_traces_folder'])
" "$CONFIG")
  mkdir -p "outputs/$OUTDIR"
  python -m chopper.profile.collect --gpu-telemetry --cpu-telemetry --telemetry-off 0.01 --output-dir "outputs/$OUTDIR" -- \
  ./runner/primus-cli container --image rocm/primus:v26.2 --volume "$PWD":/workspace/Primus -- train pretrain --config "$CONFIG"
done
