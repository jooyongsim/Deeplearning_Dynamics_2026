export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for k in mlp diff; do
  for n in 25 50 100 180 360; do ../venv/bin/python -u scale2.py $k $n 30000 512 > sc4_${k}_$n.txt 2>&1 & done
  wait
done
grep -h RESULT sc4_*.txt
