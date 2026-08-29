from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


def _zmq_addr(name: str, offset: int) -> str:
    """Channel address for the ``name`` IPC channel.

    POSIX uses ``ipc:///tmp/<name>.pid=<pid>``. Windows pyzmq ships a libzmq
    without ipc:// support (bind raises ``ZMQError: Protocol not supported``),
    so channels fall back to loopback TCP with ports derived from this
    instance's PID: ``20000 + pid % 20000 + offset``. The offset keeps the five
    channels (backend/detokenizer/broadcast/frontend/tokenizer) distinct; the
    workers receive the parent-computed string, so every process shares one
    namespace.
    """
    import os

    if os.name != "nt":
        return f"ipc:///tmp/{name}.pid={os.getpid()}"
    return f"tcp://127.0.0.1:{20000 + os.getpid() % 20000 + offset}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return _zmq_addr("freetoken_0", 0)

    @property
    def zmq_detokenizer_addr(self) -> str:
        return _zmq_addr("freetoken_1", 1)

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return _zmq_addr("freetoken_2", 2)

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
