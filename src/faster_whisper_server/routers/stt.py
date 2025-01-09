from __future__ import annotations

import asyncio
from io import BytesIO
import logging
import time
from typing import TYPE_CHECKING, Annotated

import av.error
from fastapi import (
    APIRouter,
    Depends,
    Form,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.websockets import WebSocketState
from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import BatchedInferencePipeline
from faster_whisper.vad import VadOptions, get_speech_timestamps
from numpy.typing import NDArray
from pydantic import AfterValidator, Field
import numpy as np

from faster_whisper_server.api_models import (
    DEFAULT_TIMESTAMP_GRANULARITIES,
    TIMESTAMP_GRANULARITIES_COMBINATIONS,
    CreateTranscriptionResponseJson,
    CreateTranscriptionResponseVerboseJson,
    TimestampGranularities,
    TranscriptionSegment,
)
from faster_whisper_server.config import (
    SAMPLES_PER_SECOND,
    Language,
    ResponseFormat,
    Task,
)
from faster_whisper_server.dependencies import (
    ConfigDependency,
    ModelManagerDependency,
    get_config,
)
from faster_whisper_server.text_utils import (
    segments_to_srt,
    segments_to_text,
    segments_to_vtt,
)


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from faster_whisper.transcribe import TranscriptionInfo


logger = logging.getLogger(__name__)

router = APIRouter()


# TODO: test async vs sync performance
async def audio_file_dependency(
    file: Annotated[UploadFile, Form()],
) -> NDArray[np.float32]:
    try:
        binary = await file.read()  # Use async read
        bytes = BytesIO(binary)
        bytes.seek(0)
        # Move the numpy operations to a thread since they're CPU-bound
        audio = await asyncio.to_thread(
            lambda: np.frombuffer(bytes.getbuffer(), dtype=np.int16).astype(np.float32)
            / 32768.0
        )
    except av.error.InvalidDataError as e:
        print(f"Error {e}, {file}")
        raise HTTPException(
            status_code=415,
            detail="Failed to decode audio. The provided file type is not supported.",
        ) from e
    except av.error.ValueError as e:
        raise HTTPException(
            status_code=400,
            # TODO: list supported file types
            detail="Failed to decode audio. The provided file is likely empty.",
        ) from e
    except Exception as e:
        logger.exception(
            "Failed to decode audio. This is likely a bug. Please create an issue at https://github.com/fedirz/faster-whisper-server/issues/new."
        )
        raise HTTPException(status_code=500, detail="Failed to decode audio.") from e
    else:
        return audio  # pyright: ignore reportReturnType


AudioFileDependency = Annotated[NDArray[np.float32], Depends(audio_file_dependency)]


def segments_to_response(
    segments: Iterable[TranscriptionSegment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> Response:
    segments = list(segments)
    match response_format:
        case ResponseFormat.TEXT:
            return Response(segments_to_text(segments), media_type="text/plain")
        case ResponseFormat.JSON:
            return Response(
                CreateTranscriptionResponseJson.from_segments(
                    segments
                ).model_dump_json(),
                media_type="application/json",
            )
        case ResponseFormat.VERBOSE_JSON:
            return Response(
                CreateTranscriptionResponseVerboseJson.from_segments(
                    segments, transcription_info
                ).model_dump_json(),
                media_type="application/json",
            )
        case ResponseFormat.VTT:
            return Response(
                "".join(
                    segments_to_vtt(segment, i) for i, segment in enumerate(segments)
                ),
                media_type="text/vtt",
            )
        case ResponseFormat.SRT:
            return Response(
                "".join(
                    segments_to_srt(segment, i) for i, segment in enumerate(segments)
                ),
                media_type="text/plain",
            )


def handle_default_openai_model(model_name: str) -> str:
    """Exists because some callers may not be able override the default("whisper-1") model name.

    For example, https://github.com/open-webui/open-webui/issues/2248#issuecomment-2162997623.
    """
    config = get_config()  # HACK
    if model_name == "whisper-1":
        logger.info(
            f"{model_name} is not a valid model name. Using {config.whisper.model} instead."
        )
        return config.whisper.model
    return model_name


ModelName = Annotated[
    str,
    AfterValidator(handle_default_openai_model),
    Field(
        description="The ID of the model. You can get a list of available models by calling `/v1/models`.",
        examples=[
            "Systran/faster-distil-whisper-large-v3",
            "bofenghuang/whisper-large-v2-cv11-french-ct2",
        ],
    ),
]


# HACK: Since Form() doesn't support `alias`, we need to use a workaround.
async def get_timestamp_granularities(request: Request) -> TimestampGranularities:
    form = await request.form()
    if form.get("timestamp_granularities[]") is None:
        return DEFAULT_TIMESTAMP_GRANULARITIES
    timestamp_granularities = form.getlist("timestamp_granularities[]")
    assert (
        timestamp_granularities in TIMESTAMP_GRANULARITIES_COMBINATIONS
    ), f"{timestamp_granularities} is not a valid value for `timestamp_granularities[]`."
    return timestamp_granularities


async def transcribe_with_model(
    model_manager,
    audio,
    model,
    language,
    prompt,
    # timestamp_granularities,
    temperature,
    vad_filter,
    hotwords,
    config,
):
    async with await model_manager.load_model(model) as whisper:
        whisper_model = (
            BatchedInferencePipeline(model=whisper)
            if config.whisper.use_batched_mode
            else whisper
        )
        segments, transcription_info = whisper_model.transcribe(
            audio,
            task=Task.TRANSCRIBE,
            language=language,
            initial_prompt=prompt,
            # word_timestamps="word" in timestamp_granularities,
            temperature=temperature,
            vad_filter=vad_filter,
            hotwords=hotwords,
        )
        return segments, transcription_info


# https://platform.openai.com/docs/api-reference/audio/createTranscription
# https://github.com/openai/openai-openapi/blob/master/openapi.yaml#L8915
@router.post(
    "/v1/audio/transcriptions",
    response_model=str
    | CreateTranscriptionResponseJson
    | CreateTranscriptionResponseVerboseJson,
)
async def transcribe_file(
    config: ConfigDependency,
    model_manager: ModelManagerDependency,
    request: Request,
    audio: AudioFileDependency,
    model: Annotated[ModelName | None, Form()] = None,
    language: Annotated[Language | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat | None, Form()] = None,
    temperature: Annotated[float, Form()] = 0.0,
    # timestamp_granularities: Annotated[
    #     TimestampGranularities,
    #     # WARN: `alias` doesn't actually work.
    #     Form(alias="timestamp_granularities[]"),
    # ] = ["segment"],
    stream: Annotated[bool, Form()] = False,
    hotwords: Annotated[str | None, Form()] = None,
    vad_filter: Annotated[bool, Form()] = False,
) -> Response | StreamingResponse:
    if model is None:
        model = config.whisper.model
    if language is None:
        language = config.default_language
    if response_format is None:
        response_format = config.default_response_format
    # timestamp_granularities = await get_timestamp_granularities(request)
    # if (
    #     timestamp_granularities != DEFAULT_TIMESTAMP_GRANULARITIES
    #     and response_format != ResponseFormat.VERBOSE_JSON
    # ):
    #     logger.warning(
    #         "It only makes sense to provide `timestamp_granularities[]` when `response_format` is set to `verbose_json`. See https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities."  # noqa: E501
    #     )
    start = time.perf_counter()
    segments, transcription_info = await transcribe_with_model(
        model_manager,
        audio,
        model,
        language,
        prompt,
        timestamp_granularities,
        temperature,
        vad_filter,
        hotwords,
        config,
    )
    segments = TranscriptionSegment.from_faster_whisper_segments(segments)
    end = time.perf_counter()
    logger.info(f"Transcribed in {end - start:.2f} seconds.")

    return segments_to_response(segments, transcription_info, response_format)
