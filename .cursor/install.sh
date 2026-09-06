#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for Odysseus.
#
# Runs after the repository is checked out. Prepares a Python virtualenv,
# installs the pinned dependencies, and initializes local data/db state via
# the project's own setup.py. Safe to re-run: every step no-ops when already
# satisfied.
set -euo pipefail

cd "$(dirname "$0")/.."

# 1. Ensure the stdlib venv module is usable. The default Cloud Agent image
#    ships Python 3.12 but not the python3.12-venv package (no ensurepip),
#    which the native/manual install path in README.md relies on.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  echo "[install] Installing python3-venv (ensurepip missing)..."
  sudo apt-get update -qq
  sudo apt-get install -y python3-venv "python3.12-venv" >/dev/null 2>&1 \
    || sudo apt-get install -y python3-venv >/dev/null 2>&1
fi

# 2. Create the virtualenv if it does not already exist.
if [ ! -x "venv/bin/python" ]; then
  echo "[install] Creating virtualenv..."
  python3 -m venv venv
fi

# 3. Install/refresh Python dependencies (pinned by requirements.txt).
echo "[install] Installing Python dependencies..."
./venv/bin/pip install --upgrade pip >/dev/null
./venv/bin/pip install -r requirements.txt

# 4. Initialize data dirs, SQLite DB, .env, and a first admin user.
#    Non-interactive: setup.py generates a random admin password (loopback
#    auth is bypassed for local dev via LOCALHOST_BYPASS in environment.json,
#    so no credential needs to be committed). Idempotent: skips existing files.
echo "[install] Running project setup..."
ODYSSEUS_SKIP_ADMIN_PROMPT=1 ./venv/bin/python setup.py

echo "[install] Done."
