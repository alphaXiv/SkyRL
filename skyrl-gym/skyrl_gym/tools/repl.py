import io
import signal
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    error: Optional[str]
    namespace_keys: List[str]


class _REPLTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _REPLTimeout("Code execution timed out")


def _can_use_sigalrm() -> bool:
    """SIGALRM is only usable from the main thread on Unix."""
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()


class PersistentREPL:
    """
    A persistent Python REPL that maintains state (variables) across
    multiple exec() calls. Used by RLMEnv to give the model a stateful
    programming environment it can interact with across turns.
    """

    BLOCKED_IMPORTS = frozenset({
        "subprocess", "shutil", "requests", "urllib",
        "http", "socket", "ftplib", "smtplib",
    })

    def __init__(self, namespace: Optional[Dict[str, Any]] = None, timeout: float = 15.0):
        self.namespace: Dict[str, Any] = namespace if namespace is not None else {}
        self.timeout = timeout
        self.namespace["__builtins__"] = __builtins__

    def execute(self, code: str) -> REPLResult:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        error = None

        if _can_use_sigalrm():
            error = self._execute_with_sigalrm(code, stdout_buf, stderr_buf)
        else:
            error = self._execute_with_thread_timeout(code, stdout_buf, stderr_buf)

        user_keys = [k for k in self.namespace if not k.startswith("__")]
        return REPLResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            error=error,
            namespace_keys=user_keys,
        )

    def _execute_with_sigalrm(self, code: str, stdout_buf: io.StringIO, stderr_buf: io.StringIO) -> Optional[str]:
        old_alarm = None
        try:
            old_alarm = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(self.timeout))
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, self.namespace)
        except _REPLTimeout:
            return f"Timeout after {int(self.timeout)} seconds"
        except Exception:
            return traceback.format_exc()
        finally:
            signal.alarm(0)
            if old_alarm is not None:
                signal.signal(signal.SIGALRM, old_alarm)
        return None

    def _execute_with_thread_timeout(self, code: str, stdout_buf: io.StringIO, stderr_buf: io.StringIO) -> Optional[str]:
        """Fallback when SIGALRM is unavailable (e.g. non-main thread in Ray workers)."""
        result: dict[str, Optional[str]] = {"error": None}

        def _run():
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(code, self.namespace)
            except Exception:
                result["error"] = traceback.format_exc()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=self.timeout)
        if t.is_alive():
            return f"Timeout after {int(self.timeout)} seconds"
        return result["error"]

    def summarize_namespace(self, max_repr_len: int = 80) -> str:
        """Return a compact summary of user-visible variables in the namespace."""
        parts = []
        for key in sorted(self.namespace):
            if key.startswith("__"):
                continue
            val = self.namespace[key]
            if callable(val) and not isinstance(val, type):
                parts.append(f"{key} (function)")
            elif isinstance(val, str):
                parts.append(f"{key} (str, {len(val)} chars)")
            elif isinstance(val, (list, tuple)):
                parts.append(f"{key} ({type(val).__name__}, {len(val)} items)")
            elif isinstance(val, dict):
                parts.append(f"{key} (dict, {len(val)} keys)")
            else:
                r = repr(val)
                if len(r) > max_repr_len:
                    r = r[:max_repr_len] + "..."
                parts.append(f"{key} = {r}")
        return ", ".join(parts)
