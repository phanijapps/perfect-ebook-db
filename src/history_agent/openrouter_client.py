"""OpenRouter chat model wrapper."""

from typing import Optional
from langchain_community.chat_models import ChatOpenAI
import os


class ChatOpenRouter(ChatOpenAI):
    """ChatOpenAI wrapper configured for OpenRouter."""

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self, openai_api_key: Optional[str] = None, **kwargs) -> None:
        openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key,
            **kwargs,
        )

