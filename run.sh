#!/bin/bash
# RoutellM Launcher — starts both backend API and Streamlit frontend

set -e

echo "=== RoutellM Launcher ==="
echo "Starting backend API on port 8000..."
uvicorn app.api_server:app --host 0.0.0.0 --port 8000 --log-level warning &
BACKEND_PID=$!

sleep 2

echo "Starting Streamlit frontend on port 8501..."
streamlit run streamlit_app/Home.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!

echo ""
echo "Backend API:  http://localhost:8000"
echo "Frontend:     http://localhost:8501"
echo ""

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT SIGINT SIGTERM

wait
