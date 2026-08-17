#!/usr/bin/env bash
# Sealed launcher for the Karpathy v2 baseline.
#
# Why this file exists.  V2 scrubs PATH and every loader variable from the arm
# environment, and rejects an arm that tries to set them back -- correctly, since
# an arm that can rewrite the loader can execute bytes the manifest never sealed.
# But torch.compile drives Triton, Triton shells out to `/bin/gcc`, and gcc needs
# PATH to find `as` and `ld`.  With PATH unset the whole compile dies at
# `collect2: fatal error: cannot find 'ld'`.
#
# The protocol's answer to exactly this case is a sealed launcher entered through
# an explicitly trusted system shell (`/bin/bash launch.sh`).  This file is a
# sealed code binding, so its bytes are digest-bound in the immutable manifest and
# the environment it constructs is part of the record rather than an untracked
# side effect of the worker's shell.
set -euo pipefail

# A fixed, minimal system PATH.  Not inherited -- stated, so it is reproducible
# from the sealed bytes alone.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# The interpreter lives at the resource workdir root, which is always the cwd the
# execution service launches from.  Resolve it before cd'ing to the code tree.
VENV_PYTHON="${PWD}/.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "sealed launcher: no interpreter at ${VENV_PYTHON}" >&2
  exit 78
fi

# Promotion materializes control and candidate code trees under separate roots, so
# this script must run train.py from beside itself rather than from the workdir.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep compile and scratch state inside the code tree.  Four GPU slots compile
# concurrently on one host; a shared /tmp inductor cache makes them collide, and a
# per-tree cache also stays warm across runs of identical bytes on the same slot.
export TORCHINDUCTOR_CACHE_DIR="${HERE}/.cache/inductor"
export TRITON_CACHE_DIR="${HERE}/.cache/triton"
export TMPDIR="${HERE}/.cache/tmp"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${TMPDIR}"

cd "${HERE}"
exec "${VENV_PYTHON}" train.py
