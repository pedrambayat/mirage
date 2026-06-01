#!/usr/bin/env bash
# Build a mirage-local HMMER so ANARCI can number VHH variable domains without
# depending on any external conda env. Installs hmmscan to <repo>/.tools/hmmer.
# One-time bootstrap; the .tools/ dir is gitignored. Idempotent.
set -euo pipefail

HMMER_VERSION="${HMMER_VERSION:-3.4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$REPO_ROOT/.tools"
PREFIX="$TOOLS_DIR/hmmer"
BUILD_DIR="$TOOLS_DIR/build"

if [ -x "$PREFIX/bin/hmmscan" ]; then
  echo "hmmscan already installed: $PREFIX/bin/hmmscan"
  "$PREFIX/bin/hmmscan" -h | head -2
  exit 0
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
TARBALL="hmmer-${HMMER_VERSION}.tar.gz"
[ -f "$TARBALL" ] || curl -fSL "http://eddylab.org/software/hmmer/${TARBALL}" -o "$TARBALL"
rm -rf "hmmer-${HMMER_VERSION}"
tar xzf "$TARBALL"
cd "hmmer-${HMMER_VERSION}"
./configure --prefix="$PREFIX" >/tmp/hmmer_configure.log 2>&1
make -j"$(nproc)" >/tmp/hmmer_make.log 2>&1
make install >/tmp/hmmer_install.log 2>&1

echo "installed: $PREFIX/bin/hmmscan"
"$PREFIX/bin/hmmscan" -h | head -2
