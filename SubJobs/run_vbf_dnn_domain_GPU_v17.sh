cd /afs/cern.ch/work/q/qguo/Hmumu/ML_hmm_new/Domain

WORKDIR=$PWD
RUN=noBr4_simple_dnn_DAandNoDomain_20260721_allYears_GPU_v2
python3 -u scripts/vbf_dnn_domain_noBr4_v17.py \
  --only-years all \
  --use-source-year \
  --output-name "$RUN" \
  --prepare-folds-only

python3 -u scripts/vbf_dnn_domain_noBr4_v17.py \
  --only-years all \
  --use-source-year \
  --output-name "$RUN" \
  --write-condor \
  --condor-workdir "$WORKDIR" \
  --condor-log-prefix vbf_dnn_domain_allyear_GPU \
  --condor-cpus 4 \
  --condor-gpus 1 \
  --condor-memory 16GB \
  --condor-flavour nextweek \
  --mixed-precision
