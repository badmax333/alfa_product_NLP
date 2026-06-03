#!/usr/bin/env bash
# Запуск демо из корня проекта (после: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt)
set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON:-python}"
fi
exec "$PYTHON_BIN" -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
