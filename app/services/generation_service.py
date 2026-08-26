import httpx

from app.core.config import settings


class GenerationServiceError(Exception):
    """Raised when the local Ollama generation service is unavailable."""


class GenerationService:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (
            base_url
            if base_url is not None
            else settings.OLLAMA_BASE_URL
        )
        self.model = (
            model
            if model is not None
            else settings.OLLAMA_MODEL
        )
        self.timeout = (
            timeout
            if timeout is not None
            else settings.OLLAMA_TIMEOUT
        )

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                trust_env=False,
            ) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )

            response.raise_for_status()

        except httpx.HTTPError as exc:
            raise GenerationServiceError(
                "The local LLM service is unavailable."
            ) from exc

        data = response.json()

        if "response" not in data:
            raise GenerationServiceError(
                "The local LLM service returned an invalid response."
            )

        return data["response"]