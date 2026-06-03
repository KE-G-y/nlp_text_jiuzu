#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/yaoyao/anaconda3/envs/TMF_env/bin/python}"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_PORT="${BACKEND_PORT:-8003}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 解释器不存在或不可执行：$PYTHON_BIN"
  echo "可通过 PYTHON_BIN=/path/to/python ./start.sh 指定。"
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "前端依赖不存在，开始安装 npm 依赖..."
  npm --prefix "$FRONTEND_DIR" install
fi

echo "启动后端服务：http://localhost:$BACKEND_PORT"
(
  cd "$BACKEND_DIR"
  "$PYTHON_BIN" api.py
) &
BACKEND_PID=$!

echo "启动前端服务：http://localhost:$FRONTEND_PORT"
(
  cd "$FRONTEND_DIR"
  npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

echo "服务已启动："
echo "- 后端 PID: $BACKEND_PID"
echo "- 前端 PID: $FRONTEND_PID"
echo "按 Ctrl+C 停止全部服务。"

wait
