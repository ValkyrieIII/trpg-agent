#!/bin/bash
# Start both FastAPI backend and Vite frontend

cd /workspace/projects
nohup python -m uvicorn trpg_agent.api_server:app --host 0.0.0.0 --port 8000 > /app/work/logs/bypass/api.log 2>&1 &

cd /workspace/projects/web
npx vite --host 0.0.0.0 --port 5000
