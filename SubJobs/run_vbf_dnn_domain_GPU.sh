cd /afs/cern.ch/work/q/qguo/Hmumu/ML_hmm_new/Domain

source /cvmfs/sft.cern.ch/lcg/views/LCG_107_cuda/x86_64-el9-gcc11-opt/setup.sh

RUN=noBr4_simple_dnn_DAandNoDomain_20260721_allYears_gpu_v1

python3 -u scripts/vbf_dnn_domain_noBr4_v16_improved_v2.py \
  --output-name "$RUN" \
  --prepare-folds-only

python3 -u scripts/vbf_dnn_domain_noBr4_v16_improved_v2.py \
  --output-name "$RUN" \
  --write-condor \
  --condor-gpus 1 \
  --condor-cpus 4 \
  --condor-memory 16GB \
  --condor-flavour nextweek \
  --mixed-precision \
  --skip-feature-importance
