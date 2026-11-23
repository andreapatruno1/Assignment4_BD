#!/bin/bash
# Measures the bytes really exchanged between the MPI processes with the Open MPI "pml monitoring" component.
# Separate run (monitoring adds overhead): 0 warm-up, 1 repetition, n = 2048, P = 2, 3, 4, shared memory.
# Output: results/traffic/P<P>.<rank>.prof (point-to-point lines "E <from> <to> <bytes> bytes ...").
# "--mca pml ob1" is not given here: the monitoring component is itself a pml layer that wraps ob1.
# The profile files are written to a temporary directory first (MCA values must not contain spaces).
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
DATA="${MATMUL_DATA:-data}"
TMP=$(mktemp -d)
mkdir -p results/traffic
for P in 2 3 4; do
  mpirun --oversubscribe --bind-to none --mca btl self,sm -np $P \
         --mca pml_monitoring_enable 1 --mca pml_monitoring_enable_output 3 \
         --mca pml_monitoring_filename "$TMP/P$P" build/matmul4 bench 2048 0 1 "$DATA" > /dev/null
done
cp "$TMP"/*.prof results/traffic/
ls results/traffic
