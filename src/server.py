"""OpenAI-compatible server around NemotronEngine.

The model is loaded once; the decoding mode is chosen per request through the
`model` field:

    nemotron-ar   -> autoregressive decoding
    nemotron-dlm  -> block-diffusion decoding
    nemotron-mix  -> router-interleaved AR + diffusion
    nemotron      -> the server's --mode default

    uv run python -m src.server --port 8000
"""

import argparse
import threading
import time
import uuid
from typing import Any

import fastapi
import pydantic
import uvicorn

from src import engine as engine_lib
from src import router as router_lib
from src.main import DTYPES

MODEL_PREFIX = "nemotron"


class ChatRequest(pydantic.BaseModel):
    """Subset of the OpenAI chat completion request that we honor."""

    model: str = MODEL_PREFIX
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    block_length: int | None = None
    threshold: float | None = None


def _flatten_content(content: Any) -> str:
    """Collapses OpenAI content parts (list of {"type": "text"}) to a string."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [])


def _mode_from_model(model: str, default: engine_lib.Mode) -> engine_lib.Mode:
    """Maps a model name like "openai/nemotron-dlm" to a decoding mode."""
    suffix = model.rsplit("/", 1)[-1].removeprefix(MODEL_PREFIX).lstrip("-")
    if not suffix:
        return default
    if suffix not in engine_lib.MODES:
        raise fastapi.HTTPException(
            400,
            f"Unknown model {model!r}; use nemotron-ar, nemotron-dlm or "
            "nemotron-mix.",
        )
    return suffix


def create_app(
    engine: engine_lib.NemotronEngine, max_new_tokens: int
) -> fastapi.FastAPI:
    """Builds the FastAPI app.

    Args:
        engine: Loaded engine shared by all requests.
        max_new_tokens: Default generation budget when a request has none.

    Returns:
        The FastAPI application.
    """
    app = fastapi.FastAPI(title="Nemotron-Labs-Diffusion")
    # One GPU, one generation at a time.
    lock = threading.Lock()

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        names = [MODEL_PREFIX] + [
            f"{MODEL_PREFIX}-{m}" for m in engine_lib.MODES
        ]
        return {
            "object": "list",
            "data": [{"id": n, "object": "model"} for n in names],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> dict[str, Any]:
        mode = _mode_from_model(request.model, engine.mode)
        messages = [
            {"role": m["role"], "content": _flatten_content(m.get("content"))}
            for m in request.messages
        ]
        kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_completion_tokens
            or request.max_tokens
            or max_new_tokens
        }
        if request.block_length is not None:
            kwargs["block_length"] = request.block_length
        if request.threshold is not None:
            kwargs["threshold"] = request.threshold

        with lock:
            result = engine.generate(messages, mode=mode, **kwargs)
        prompt_tokens = engine.encode_chat(messages).shape[1]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"{MODEL_PREFIX}-{mode}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": (
                        "length"
                        if result.num_tokens >= kwargs["max_new_tokens"]
                        else "stop"
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": result.num_tokens,
                "total_tokens": prompt_tokens + result.num_tokens,
            },
            "nfe": result.nfe,
            "tokens_per_second": result.tokens_per_second,
        }

    return app


def main() -> None:
    """Loads the model once and serves it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=engine_lib.DEFAULT_REPO)
    parser.add_argument("--mode", default="ar", choices=engine_lib.MODES)
    parser.add_argument(
        "--router", default="entropy", choices=router_lib.ROUTERS
    )
    parser.add_argument("--entropy-threshold", type=float, default=0.5)
    parser.add_argument("--ar-chunk", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=DTYPES)
    args = parser.parse_args()

    engine = engine_lib.NemotronEngine(
        args.model,
        device=args.device,
        dtype=DTYPES[args.dtype],
        mode=args.mode,
        router=router_lib.make_router(args.router, args.entropy_threshold),
        ar_chunk=args.ar_chunk,
    )
    uvicorn.run(
        create_app(engine, args.max_new_tokens),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
