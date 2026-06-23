#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/sweep_results"
mkdir -p "${RESULTS_DIR}"

BUILD_DIR="${SCRIPT_DIR}/Build_clang"
BINARY="${BUILD_DIR}/src/benchmarksuite-main"

echo "========================================================"
echo " CLANG SWEEP — CPB 1-16 & BPS 1-16 (Powers of 2)"
echo "========================================================"

for cpb in 1 2 4 8 16; do
    for bps in 1 2 4 8 16; do
        echo ""
        echo "--------------------------------------------------------"
        echo " CLANG | CHUNKS_PER_BLOCK=${cpb} | BLOCKS_PER_STEP=${bps}"
        echo "--------------------------------------------------------"

        cmake -S . -B "${BUILD_DIR}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER=/usr/bin/clang++-23 \
            -DCMAKE_C_COMPILER=/usr/bin/clang-23 \
            -DSIMDJSON_DEVELOPMENT_CHECKS=OFF \
            -DBNCH_SWT_BENCHMARKS=ON \
            -DCHUNKS_PER_BLOCK="${cpb}" \
            -DBLOCKS_PER_STEP="${bps}"

        cmake --build "${BUILD_DIR}" --parallel $(nproc)

        result_file="${RESULTS_DIR}/clang_cpb${cpb}_bps${bps}.txt"
        echo "# COMPILER=clang CHUNKS_PER_BLOCK=${cpb} BLOCKS_PER_STEP=${bps}" > "${result_file}"
        sudo "${BINARY}" 2>&1 | grep -v "^\-\-\-\|^All correct\|^Building\|entries:\|buf size:\|\[bench\]" >> "${result_file}"

        echo "Saved: ${result_file}"
        cat "${result_file}"
    done
done

echo ""
echo "========================================================"
echo " CLANG SWEEP COMPLETE — SUMMARY"
echo "========================================================"
echo ""
printf "%-6s %-6s %-22s %-22s %-22s\n" "CPB"