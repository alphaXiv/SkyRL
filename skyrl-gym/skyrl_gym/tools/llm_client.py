class ExternalSubLLMClient:
    """
    Synchronous client for sub-LLM calls to a frozen external model.
    Uses the OpenAI-compatible chat completions API (works with vLLM, OpenAI, etc.).

    If ``base_url`` is None the official OpenAI endpoint is used and
    ``api_key`` falls back to the ``OPENAI_API_KEY`` environment variable.
    """

    def __init__(self, base_url: str | None = None, model: str = "default", api_key: str | None = None):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required for ExternalSubLLMClient: pip install openai")

        kwargs: dict = {}
        if base_url is not None:
            kwargs["base_url"] = base_url
            kwargs["api_key"] = api_key or "EMPTY"
        elif api_key is not None:
            kwargs["api_key"] = api_key
        # When neither base_url nor api_key is set, the openai client
        # reads OPENAI_API_KEY from the environment automatically.

        self.client = openai.OpenAI(**kwargs)
        self.model = model

    def query(self, prompt_str: str, max_tokens: int = 4096) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_str}],
                max_completion_tokens=max_tokens,
            )
        except Exception as e:
            raise RuntimeError(
                f"llm_query failed (model={self.model}, prompt_len={len(prompt_str)}, "
                f"{type(e).__name__}): {e}"
            ) from e
        return response.choices[0].message.content
