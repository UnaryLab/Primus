#!/usr/bin/env bash
set -eux

OUTBASE="outputs"
PYTHON="${PYTHON:-/home/mkurzyns/.venv/bin/python3}"
CONFIGS=(b1s4 b2s4 b4s4 b1s8 b2s8)

# Step 1: Process each config's traces with merge3
for cfg in "${CONFIGS[@]}"; do
    TRACEDIR="${OUTBASE}/profile_trace_${cfg}"
    PKLDIR="${OUTBASE}/pickles_${cfg}"
    mkdir -p "$PKLDIR"

    for iter_dir in "${TRACEDIR}"/iteration_*/; do
        iter_name=$(basename "$iter_dir")
        iter_pkl="${PKLDIR}/${iter_name}.pkl"
        if [ ! -f "$iter_pkl" ]; then
            echo "Processing ${cfg}/${iter_name}..."
            $PYTHON -m chopper.profile.merge -t "${iter_dir}"rank*_trace.json -o "$iter_pkl"
        fi
    done

    # Combine iterations
    COMBINED="${PKLDIR}/ts.pkl"
    if [ ! -f "$COMBINED" ]; then
        echo "Combining ${cfg}..."
        $PYTHON -m chopper.profile.merge -p "${PKLDIR}"/iteration_*.pkl -o "$COMBINED"
    fi
done

echo "Done. Pickles at:"
ls -la "${OUTBASE}"/pickles_*/ts.pkl
