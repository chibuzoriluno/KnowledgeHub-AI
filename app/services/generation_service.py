import httpx


class GenerationService:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma2:2b",
    ):
        self.base_url = base_url
        self.model = model

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=60.0, trust_env=False,) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )

        response.raise_for_status()

        data = response.json()

        return data["response"]