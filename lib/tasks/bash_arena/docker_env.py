import os
import subprocess
import json
import docker


def _get_docker_client():
    """Get Docker client, detecting Docker Desktop's socket if needed."""
    try:
        return docker.from_env()
    except docker.errors.DockerException:
        pass
    # Fall back to Docker context socket (e.g. Docker Desktop on macOS)
    result = subprocess.run(["docker", "context", "inspect"], capture_output=True, text=True)
    if result.returncode == 0:
        ctx = json.loads(result.stdout)
        host = ctx[0]["Endpoints"]["docker"]["Host"]
        return docker.DockerClient(base_url=host)
    raise docker.errors.DockerException("Cannot connect to Docker daemon")


class DockerEnvironment:
    def __init__(self, image: str, setup_script: str):
        self.image = image
        self.setup_script = setup_script
        self.client = _get_docker_client()
        self.container = None

    def start(self):
        self.container = self.client.containers.run(
            self.image, command="sleep infinity", detach=True, remove=False,
        )
        if self.setup_script:
            self.execute(self.setup_script)

    def execute(self, command: str, timeout: float = 30.0) -> tuple[str, int]:
        result = self.container.exec_run(
            ["bash", "-c", command], demux=True,
        )
        stdout = result.output[0].decode() if result.output[0] else ""
        stderr = result.output[1].decode() if result.output[1] else ""
        output = stdout + stderr
        return output, result.exit_code

    def evaluate(self, eval_script: str) -> bool:
        _, exit_code = self.execute(eval_script)
        return exit_code == 0

    def cleanup(self):
        if self.container:
            self.container.stop(timeout=5)
            self.container.remove(force=True)
            self.container = None

    async def __aenter__(self):
        self.start()
        return self

    async def __aexit__(self, *exc):
        self.cleanup()
