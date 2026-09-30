import os

from openai import OpenAI

try:  # optional: load .env from the working directory / repo root
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

DEFAULT_BASE_URL = "https://api.openai.com/v1"

_client = None


def get_client() -> OpenAI:
    """Lazily build a single OpenAI client from environment variables."""
    global _client
    if _client is None:
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError("API_KEY is not set (export API_KEY=... or put it in .env)")
        _client = OpenAI(
            base_url=os.getenv("API_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key,
        )
    return _client


def query_model(prompt, model_name, **kwargs):
    """
    Send a chat request to `model_name` and return the reply text.

    Args:
        prompt: either a plain user message (str) or a full list of
            ``{"role": ..., "content": ...}`` messages.
        model_name: model identifier understood by the endpoint.
        **kwargs: extra arguments forwarded to ``chat.completions.create``
            (e.g. temperature, max_tokens).

    Returns:
        The reply content, or the error message as a string when the request
        fails (callers treat non-parsable output as an evaluation error).
    """
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        messages = prompt
    else:
        return "Error: a prompt must be a string or a list of messages."

    try:
        response = get_client().chat.completions.create(
            model=model_name, messages=messages, **kwargs
        )
        return response.choices[0].message.content
    except Exception as e:  # noqa: BLE001 - surfaced to caller as text
        return str(e)
