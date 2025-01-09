UVICORN_HOST=0.0.0.0 UVICORN_PORT=8080 uv run uvicorn --factory faster_whisper_server.main:create_app
