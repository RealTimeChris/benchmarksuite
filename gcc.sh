#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/sweep_results"
mkdir -p "${RESULTS_DIR}"

BUILD_DIR="${SCRIPT_DIR}/Build_gcc"
BINARY="${BUILD_DIR}/src/benchmarksuite-main"

echo "========================================================"
echo " GCC SWEEP — CPB 1-16 & BPS 1-16 (Powers of 2)"
echo "========================================================"

for cpb in 1 2 4 8 16; do
    for bps in 1 2 4 8 16; do
        echo ""
        echo "--------------------------------------------------------"
        echo " GCC | CHUNKS_PER_BLOCK=${cpb} | BLOCKS_PER_STEP=${bps}"
        echo "--------------------------------------------------------"

        cmake -S . -B "${BUILD_DIR}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER=/usr/bin/g++-16 \
            -DCMAKE_C_COMPILER=/usr/bin/gcc-16 \
            -DSIMDJSON_DEVELOPMENT_CHECKS=OFF \
            -DBNCH_SWT_BENCHMARKS=ON \
            -DCHUNKS_PER_BLOCK="${cpb}" \
            -DBLOCKS_PER_STEP="${bps}" \
            > /dev/null 2>&1

        cmake --build "${BUILD_DIR}" --parallel $(nproc) > /dev/null 2>&1

        result_file="${RESULTS_DIR}/gcc_cpb${cpb}_bps${bps}.txt"
        echo "# COMPILER=gcc CHUNKS_PER_BLOCK=${cpb} BLOCKS_PER_STEP=${bps}" > "${result_file}"
        sudo "${BINARY}" 2>&1 | grep -v "^\-\-\-\|^All correct\|^Building\|entries:\|buf size:\|\[bench\]" >> "${result_file}"

        echo "Saved: ${result_file}"
        cat "${result_file}"
    done
done

echo ""
echo "========================================================"
echo " GCC SWEEP COMPLETE — SUMMARY"
echo "========================================================"
echo ""
printf "%-6s %-6s %-22s %-22s %-22s\n" "CPB" "BPS" "ASCII jsonifier" "MIXED jsonifier" "MULTIBYTE jsonifier"
echo "--------------------------------------------------------------------------------"
for cpb in 1 2 4 8 16; do
    for bps in 1 2 4 8 16; do
        result_file="${RESULTS_DIR}/gcc_cpb${cpb}_bps${bps}.txt"
        [[ -f "${result_file}" ]] || continue
        ascii_j=$(grep "jsonifier" "${result_file}" | awk -F',' 'NR==1{print $2}')
        mixed_j=$(grep "jsonifier" "${result_file}" | awk -F',' 'NR==2{print $2}')
        multi_j=$(grep "jsonifier" "${result_file}" | awk -F',' 'NR==3{print $2}')
        printf "%-6s %-6s %-22s %-22s %-22s\n" "${cpb}" "${bps}" "${ascii_j:-N/A}" "${mixed_j:-N/A}" "${multi_j:-N/A}"
    done
done

echo ""
echo "--- simdjson baseline ---"
result_file="${RESULTS_DIR}/gcc_cpb1_bps1.txt"
if [[ -f "${result_file}" ]]; then
    ascii_s=$(grep "simdjson" "${result_file}" | awk -F',' 'NR==1{print $2}')
    mixed_s=$(grep "simdjson" "${result_file}" | awk -F',' 'NR==2{print $2}')
    multi_s=$(grep "simdjson" "${result_file}" | awk -F',' 'NR==3{print $2}')
    printf "%-6s %-6s %-22s %-22s %-22s\n" "smdj" "smdj" "${ascii_s:-N/A}" "${mixed_s:-N/A}" "${multi_s:-N/A}"
else
    echo "Baseline file ${result_file} not found."
fi
echo ""
echo "Raw results in: ${RESULTS_DIR}/gcc_cpb*_bps*.txt"