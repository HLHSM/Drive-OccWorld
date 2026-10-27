#!/usr/bin/env bash
set -euo pipefail
PAPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${TEXLIVE_BIN:-}" ]]; then
  export PATH="${TEXLIVE_BIN}:${PATH}"
fi
command -v latexmk >/dev/null || { echo 'latexmk not found in PATH' >&2; exit 1; }
command -v xelatex >/dev/null || { echo 'xelatex not found in PATH' >&2; exit 1; }
cd "${PAPER_DIR}"
latexmk -g -xelatex -interaction=nonstopmode -halt-on-error main.tex > compile.log 2>&1 || {
  tail -n 60 compile.log
  exit 1
}
printf 'Compiled: %s/main.pdf\n' "${PAPER_DIR}"
