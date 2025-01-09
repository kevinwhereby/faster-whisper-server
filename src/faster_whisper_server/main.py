from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import time
import platform
import concurrent.futures
import asyncio
from typing import TYPE_CHECKING

from fastapi import (
    FastAPI,
)
from fastapi.middleware.cors import CORSMiddleware

from faster_whisper_server.dependencies import (
    get_config,
    get_model_manager,
    verify_api_key,
)
from faster_whisper_server.logger import setup_logger
from faster_whisper_server.routers.list_models import (
    router as list_models_router,
)
from faster_whisper_server.routers.misc import (
    router as misc_router,
)
from faster_whisper_server.routers.stt import (
    router as stt_router,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def create_app() -> FastAPI:
    setup_logger()

    logger = logging.getLogger(__name__)

    config = get_config()  # HACK
    logger.debug(f"Config: {config}")

    model_manager = get_model_manager()  # HACK

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        for model_name in config.preload_models:
            model_manager.load_model(model_name)
        yield

    dependencies = []
    if config.api_key is not None:
        dependencies.append(verify_api_key)

    app = FastAPI(lifespan=lifespan, dependencies=dependencies)

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        logger.info(
            "Time took to process the request and return response is {} sec".format(
                time.time() - start_time
            )
        )
        return response

    app.include_router(stt_router)
    app.include_router(list_models_router)
    app.include_router(misc_router)

    cpu_count = os.cpu_count() or 1
    thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=cpu_count * 2,  # 16 threads for 8 CPUs
        thread_name_prefix="whisper_worker",
    )

    loop = asyncio.get_event_loop()
    loop.set_default_executor(thread_pool)

    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if config.enable_ui:
        import gradio as gr

        from faster_whisper_server.gradio_app import create_gradio_demo

        app = gr.mount_gradio_app(app, create_gradio_demo(config), path="/")

    return app
