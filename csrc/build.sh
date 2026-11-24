#!/bin/bash
# Build the _inkling_sm80_cuda extension for sm_80. Run it inside the vllm/vllm-openai:inkling-2-mini image (it has
# nvcc, PyTorch and Python 3.14); run.sh does this on first start and keeps the result in its cache folder.
#   OUT_DIR=/where/to/put/it bash csrc/build.sh        (default: the site/ folder next to this one)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/../site}"
mkdir -p "${OUT_DIR}"
PYINC="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
TINC="$(python3 -c 'import torch, os; p=os.path.join(os.path.dirname(torch.__file__), "include"); print(f"-I{p} -I{p}/torch/csrc/api/include")')"
TLIB="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
EXT_SUFFIX="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUT="${OUT_DIR}/_inkling_sm80_cuda${EXT_SUFFIX}"
rm -f "${OUT_DIR}"/_inkling_sm80_cuda*.so
nvcc -O3 --use_fast_math -std=c++20 -arch=sm_80 -shared -Xcompiler -fPIC \
  -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_inkling_sm80_cuda \
  -I"${PYINC}" ${TINC} -L"${TLIB}" -ltorch -ltorch_python -lc10 -lc10_cuda \
  "${SCRIPT_DIR}/inkling_sm80_cuda.cu" -o "${OUT}"
chmod 755 "${OUT}"
echo "Built ${OUT}"
