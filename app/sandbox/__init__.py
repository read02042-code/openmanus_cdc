"""
Docker Sandbox Module

Provides secure containerized execution environment with resource limits
and isolation for running untrusted code.
"""

from app.sandbox.client import (
    BaseSandboxClient,
    LocalSandboxClient,
    create_sandbox_client,
)
from app.sandbox.core.exceptions import (
    SandboxError,
    SandboxResourceError,
    SandboxTimeoutError,
)

try:
    from app.sandbox.core.sandbox import DockerSandbox
except Exception:
    DockerSandbox = None  # type: ignore[assignment]
try:
    from app.sandbox.core.manager import SandboxManager
except Exception:
    SandboxManager = None  # type: ignore[assignment]


__all__ = [
    "DockerSandbox",
    "SandboxManager",
    "BaseSandboxClient",
    "LocalSandboxClient",
    "create_sandbox_client",
    "SandboxError",
    "SandboxTimeoutError",
    "SandboxResourceError",
]
