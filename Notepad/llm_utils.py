from __future__ import annotations

import json
import os
import urllib.request
from typing import Generator, List, Tuple

# NOTE: llama-cpp / llama-cpp-agent imports are intentionally optional/lazy.
# This module can run in "Ollama-only" mode without these dependencies.

try:
    from llama_cpp_agent import LlamaCppAgent
    from llama_cpp_agent.providers import LlamaCppPythonProvider
    from llama_cpp_agent.chat_history import BasicChatHistory
    from llama_cpp_agent.chat_history.messages import Roles
    from llama_cpp_agent.messages_formatter import MessagesFormatter, PromptMarkers
    _HAS_LLAMA_CPP_AGENT = True
except ModuleNotFoundError:  # pragma: no cover
    LlamaCppAgent = None  # type: ignore[assignment]
    LlamaCppPythonProvider = None  # type: ignore[assignment]
    BasicChatHistory = None  # type: ignore[assignment]
    Roles = None  # type: ignore[assignment]
    MessagesFormatter = None  # type: ignore[assignment]
    PromptMarkers = None  # type: ignore[assignment]
    _HAS_LLAMA_CPP_AGENT = False

__all__ = ["respond"]

# ───────────────────────── Gemma-3 prompt markers ──────────────────────────
if _HAS_LLAMA_CPP_AGENT:
    _gemma_3_prompt_markers = {
        Roles.system:    PromptMarkers("", "\n"),
        Roles.user:      PromptMarkers("<start_of_turn>user\n",  "<end_of_turn>\n"),
        Roles.assistant: PromptMarkers("<start_of_turn>model\n", "<end_of_turn>\n"),
        Roles.tool:      PromptMarkers("", ""),
    }
    _gemma_3_formatter = MessagesFormatter(
        pre_prompt="",
        prompt_markers=_gemma_3_prompt_markers,
        include_sys_prompt_in_first_user_message=True,
        default_stop_sequences=["<end_of_turn>", "<start_of_turn>"],
        strip_prompt=False,
        bos_token="<bos>",
        eos_token="<eos>",
    )
else:
    _gemma_3_prompt_markers = {}
    _gemma_3_formatter = None


_llm = None  # cached llama-cpp model
_llm_model_path: str | None = None


def _lazy_load_model(model_path: str):
    """Load (or return cached) GGUF model from *model_path*."""
    global _llm, _llm_model_path

    if _llm and _llm_model_path == model_path:
        return _llm

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Lazy import so Ollama-only users don't need llama-cpp on import.
    from llama_cpp import Llama  # local import intentional

    _llm = Llama(
        model_path=model_path,
        flash_attn=False,
        n_gpu_layers=0,
        n_batch=8,
        n_ctx=102_400,
        n_threads=8,
        n_threads_batch=8,
    )
    _llm_model_path = model_path
    return _llm


def _is_ollama_model(model: str) -> bool:
    """Heuristic: treat *model* as Ollama if it's not a file path."""
    if model.startswith("ollama:"):
        return True
    if model.lower().endswith(".gguf"):
        return False
    if os.path.exists(model):
        return False
    # Common Ollama identifiers include tags like "llama3:8b"
    return True


def _ollama_chat_stream(
    *,
    model: str,
    message: str,
    history: List[Tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
) -> Generator[str, None, None]:
    """Stream tokens from Ollama's /api/chat endpoint."""
    model_name = model.split("ollama:", 1)[-1]
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{host}/api/chat"

    messages: list[dict] = [{"role": "system", "content": system_message}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "num_predict": max_tokens,  # output tokens
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    full = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            # NDJSON stream: one JSON per line
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if obj.get("error"):
                    raise RuntimeError(obj["error"])

                chunk = (
                    (obj.get("message") or {}).get("content")
                    or obj.get("response")
                    or ""
                )
                if chunk:
                    full += chunk
                    yield full

                if obj.get("done") is True:
                    break
    except Exception as exc:
        yield f"[Ollama Error] {exc}\n"


def respond(
    message: str,
    history: List[Tuple[str, str]],
    *,
    model: str | None = None,
    system_message: str = "You are a helpful assistant.",
    max_tokens: int = 102_400,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
):
    chosen = model or "gemma-3-1b-it-Q4_K_M.gguf"  # default

    # ── Ollama path ──
    if _is_ollama_model(chosen):
        yield from _ollama_chat_stream(
            model=chosen,
            message=message,
            history=history,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
        )
        return

    # ── llama-cpp GGUF path ──
    if not _HAS_LLAMA_CPP_AGENT:
        yield (
            "[Error] llama_cpp_agent is not installed, but a .gguf model was selected.\n"
            "Install dependencies from requirements.txt, or select an Ollama model name (e.g., llama3:8b).\n"
        )
        return

    llm = _lazy_load_model(chosen)
    provider = LlamaCppPythonProvider(llm)
    agent = LlamaCppAgent(
        provider,
        system_prompt=system_message,
        custom_messages_formatter=_gemma_3_formatter,
        debug_output=False,
    )

    settings = provider.get_provider_default_settings()
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p
    settings.max_tokens = max_tokens
    settings.repeat_penalty = repeat_penalty
    settings.stream = True

    chat_hist = BasicChatHistory()
    for user_msg, assistant_msg in history:
        chat_hist.add_message({"role": Roles.user, "content": user_msg})
        chat_hist.add_message({"role": Roles.assistant, "content": assistant_msg})

    stream = agent.get_chat_response(
        message,
        llm_sampling_settings=settings,
        chat_history=chat_hist,
        returns_streaming_generator=True,
        print_output=False,
    )

    full = ""
    try:
        for tok in stream:
            full += tok
            yield full
    except Exception as exc:
        yield f"[Error] {exc}\n"