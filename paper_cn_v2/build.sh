#!/usr/bin/env bash
set -euo pipefail
PAPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEXLIVE_BIN="${TEXLIVE_BIN:-/data/HL/texlive/2026/bin/x86_64-linux}"
export PATH="${TEXLIVE_BIN}:${PATH}"
cd "${PAPER_DIR}"
latexmk -g -xelatex -interaction=nonstopmode -halt-on-error main.tex > compile.log 2>&1 || {
  tail -n 60 compile.log
  exit 1
}
printf 'Compiled: %s/main.pdf\n' "${PAPER_DIR}"
