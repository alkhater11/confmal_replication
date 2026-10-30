#!/bin/bash
# Reproduce every number, table and figure of the revised manuscript.
# Two CPU cores are assumed; jobs run two at a time.  Total wall-clock time on
# a two-core machine is about half an hour.
#
#   KRONODROID_DIR  directory holding the four extracted KronoDroid CSV files
#                   (default: ./kd/extracted)
set -e
# Single-threaded numerical libraries: avoids oversubscription when two jobs
# share two cores and makes results independent of the thread count.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
ROOT="$(pwd)"
mkdir -p logs
cd code

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  python3 build_dataset.py real     > ../logs/build_real.log 2>&1
  python3 build_dataset.py emu      > ../logs/build_emu.log 2>&1
  python3 build_dataset.py overlap  > ../logs/build_overlap.log 2>&1
fi
python3 test_conformal_lib.py       > ../logs/tests.log 2>&1

python3 run_audit.py real RandomForest > ../logs/audit_real_RandomForest.log 2>&1 &
python3 run_audit.py real HistGB       > ../logs/audit_real_HistGB.log 2>&1 &
wait
python3 run_audit.py real LogReg       > ../logs/audit_real_LogReg.log 2>&1 &
python3 run_audit.py emu RandomForest  > ../logs/audit_emu_RandomForest.log 2>&1 &
wait
python3 run_audit.py emu HistGB        > ../logs/audit_emu_HistGB.log 2>&1 &
python3 run_audit.py emu LogReg        > ../logs/audit_emu_LogReg.log 2>&1 &
wait
python3 run_enbpi.py real              > ../logs/enbpi_real.log 2>&1 &
python3 run_enbpi.py emu               > ../logs/enbpi_emu.log 2>&1 &
wait
python3 run_diagnostic.py real         > ../logs/diagnostic_real.log 2>&1 &
python3 run_diagnostic.py emu          > ../logs/diagnostic_emu.log 2>&1 &
wait
python3 run_family.py real             > ../logs/family_real.log 2>&1 &
python3 run_feature_ablation.py real   > ../logs/ablation_real.log 2>&1 &
wait
python3 aci_truncation.py real         > ../logs/aci_truncation_real.log 2>&1
echo EXPERIMENTS_DONE > ../logs/experiments_done.txt
if [ "${SKIP_ANALYSIS:-0}" != "1" ]; then
  python3 analyze.py                   > ../logs/analyze.log 2>&1
  python3 make_numbers.py              > ../logs/numbers.log 2>&1
  python3 make_tables.py               > ../logs/tables.log 2>&1
  python3 make_figures.py              > ../logs/figures.log 2>&1
  python3 verify_numbers.py            > ../logs/verify.log 2>&1
fi
echo ALL_DONE > ../logs/all_done.txt
