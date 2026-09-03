import os
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen


class OllamaService:
    """Manages the lifecycle of a local Ollama server process."""

    def __init__(
        self,
        models_dir: str = "/home/gchmilew/42Cursus/ollama",
        host: str = "http://localhost:11434",
        timeout: int = 15,
    ) -> None:
        self.models_dir = models_dir
        self.host = host
        self.timeout = timeout
        self.process: subprocess.Popen | None = None

    def is_alive(self) -> bool:
        """Check if the Ollama service is reachable and ready."""
        try:
            with urlopen(f"{self.host}/api/tags", timeout=1) as response:
                return bool(response.status == 200)
        except (URLError, TimeoutError, ConnectionRefusedError):
            return False

    def start(self) -> None:
        """Start the background daemon
          with isolated models and wait for readiness."""
        if self.is_alive():
            return

        env = os.environ.copy()
        env["OLLAMA_MODELS"] = self.models_dir

        self.process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()
        while not self.is_alive():
            if time.time() - start_time > self.timeout:
                raise TimeoutError(
                    f"Ollama server failed to start within {self.timeout}s."
                )
            time.sleep(0.5)
