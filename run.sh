# UVICORN_HOST=0.0.0.0 UVICORN_PORT=8080 uv run uvicorn --factory faster_whisper_server.main:create_app

uv run uvicorn --factory faster_whisper_server.main:create_app \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 2 \
