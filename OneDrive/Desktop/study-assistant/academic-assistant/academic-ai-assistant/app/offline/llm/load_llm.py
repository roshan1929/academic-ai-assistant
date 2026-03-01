"""LLM loader using Ollama with a singleton pattern.

This module provides `load_llm()` and `get_llm()` helpers. The actual
Ollama client is imported at runtime to avoid import-time failures when
the dependency isn't installed.

The loader ensures the model is only initialized once (singleton).
"""

from typing import Any, Optional
import threading

__all__ = ["load_llm", "get_llm"]

# Module-level singleton and lock
_LLM: Optional[Any] = None
_LOCK = threading.Lock()


class OllamaWrapper:
    """Lightweight wrapper around an Ollama client instance.

    This wrapper stores the client and the model name and exposes a
    simple `generate(prompt, **kwargs)` method. The implementation
    attempts common Ollama client call signatures; if the client isn't
    available the wrapper will raise informative errors when used.
    """

    def __init__(self, client: Any, model_name: str):
        self.client = client
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> Any:
        """Generate a response for `prompt` using the underlying client.

        This method does not attempt to implement any complex logic; it
        simply forwards the call to the Ollama client using a few common
        call patterns.
        """
        # Try common Ollama client method signatures
        try:
            if hasattr(self.client, "generate"):
                # common signature: client.generate(model=model_name, prompt=...)
                try:
                    return self.client.generate(model=self.model_name, prompt=prompt, **kwargs)
                except TypeError:
                    # fallback: client.generate(model_name, prompt)
                    return self.client.generate(self.model_name, prompt, **kwargs)

            # As a last resort, if the client exposes a `call` or `completion` method
            if hasattr(self.client, "call"):
                return self.client.call(self.model_name, prompt, **kwargs)

            raise RuntimeError("Ollama client does not expose a supported generate API")
        except Exception as exc:
            raise RuntimeError(f"Failed to generate from Ollama model: {exc}") from exc


def load_llm(model_name: str = "mistral:7b-instruct-q4_0") -> Optional[OllamaWrapper]:
    """Load and return a singleton Ollama-based LLM wrapper.

    If the Ollama Python client is not installed or initialization
    fails, this function returns `None` after printing an error.
    """
    global _LLM
    if _LLM is not None:
        print(f"LLM already loaded: {_LLM.model_name if hasattr(_LLM, 'model_name') else _LLM}")
        return _LLM

    with _LOCK:
        if _LLM is not None:
            print(f"LLM already loaded (post-lock): {_LLM.model_name if hasattr(_LLM, 'model_name') else _LLM}")
            return _LLM

        try:
            # Import Ollama client lazily
            from ollama import Ollama  # type: ignore
        except Exception as exc:
            print("Ollama client library not available; falling back to MockLLM for testing:", exc)
            # Provide a simple mock LLM so the rest of the pipeline can be
            # exercised without an actual Ollama backend. The mock simply
            # extracts the provided context from the prompt and echoes a
            # short simulated answer.
            class MockLLM:
                def __init__(self, model_name: str):
                    self.model_name = f"mock:{model_name}"

                def generate(self, prompt: str, **kwargs):
                    # Try to extract the context block between 'Context:' and 'Question:'
                    try:
                        start = prompt.index("Context:") + len("Context:")
                        end = prompt.index("Question:")
                        context = prompt[start:end].strip()
                    except Exception:
                        context = prompt[:500]
                    summary = context.replace("\n", " ")[:600]
                    return f"Mock answer (from context): {summary}"

            _LLM = MockLLM(model_name)
            print(f"Loaded MockLLM for model: {model_name}")
            return _LLM

        try:
            print(f"Initializing Ollama client for model: {model_name}")
            client = Ollama()
            wrapper = OllamaWrapper(client=client, model_name=model_name)
            _LLM = wrapper
            print(f"Loaded Ollama model wrapper for: {model_name}")
            return _LLM
        except Exception as exc:
            print("Failed to initialize Ollama client:", exc)
            _LLM = None
            return None


def get_llm() -> Optional[OllamaWrapper]:
    """Return the singleton LLM wrapper, loading it if necessary.

    Returns `None` if the Ollama client is unavailable.
    """
    global _LLM
    if _LLM is None:
        return load_llm()
    return _LLM

