"""
Sandboxed code-execution checker for the MBPP programming OOD eval.

Runs a model-generated Python solution against the reference unit tests
(optional `test_setup_code` + candidate + `test_list` asserts) in a subprocess
with a wall-clock timeout and CPU rlimit.

SECURITY NOTE: this executes model-generated code with OS-level subprocess
isolation only (timeout + rlimits + temp cwd). Standard research practice for
MBPP; do not point it at untrusted third-party outputs.
"""

import os
import re
import resource
import subprocess
import sys
import tempfile

_COMMON_PREAMBLE = (
    "import math\n"
    "import re\n"
    "import sys\n"
    "from typing import List, Dict, Tuple, Optional, Any\n"
)

_DEF_RE = re.compile(r"^\s*(def|class)\s", re.MULTILINE)


def strip_code_fences(text: str) -> str:
    """Extract python code from an <answer> payload that may use ``` fences or prose."""
    text = text.strip()
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if blocks:
        code_blocks = [b for b in blocks if _DEF_RE.search(b)]
        chosen = code_blocks if code_blocks else blocks
        return "\n\n".join(b.strip() for b in chosen)
    return text


def _limits(timeout_s: int):
    def fn():
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_s + 2, timeout_s + 2))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    return fn


def _run_program(program: str, timeout_s: int) -> bool:
    with tempfile.TemporaryDirectory(prefix="code_exec_") as td:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", program],
                cwd=td,
                capture_output=True,
                timeout=timeout_s,
                preexec_fn=_limits(timeout_s),
                env={"PATH": os.environ.get("PATH", ""), "HOME": td,
                     "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"},
            )
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False


def run_code_tests(candidate: str, source: str, test_list=None,
                   test_setup_code: str = "", timeout_s: int = 10, **_) -> float:
    """Return 1.0 if the candidate passes the reference tests, else 0.0."""
    code = strip_code_fences(candidate or "")
    if not code.strip() or not _DEF_RE.search(code):
        return 0.0
    if source != "mbpp" or not test_list:
        return 0.0
    setup = (test_setup_code or "").strip()
    asserts = "\n".join(test_list)
    program = f"{_COMMON_PREAMBLE}\n{setup}\n\n{code}\n\n{asserts}\n"
    return 1.0 if _run_program(program, timeout_s) else 0.0


if __name__ == "__main__":
    # Smoke test with trivial known-good/known-bad candidates.
    good = "def add(a, b):\n    return a + b"
    bad = "def add(a, b):\n    return a - b"
    hang = "def add(a, b):\n    while True:\n        pass"
    assert run_code_tests(good, "mbpp", test_list=["assert add(1,2)==3"]) == 1.0
    assert run_code_tests(bad, "mbpp", test_list=["assert add(1,2)==3"]) == 0.0
    assert run_code_tests(hang, "mbpp", test_list=["assert add(1,2)==3"], timeout_s=3) == 0.0
    assert run_code_tests("```python\n" + good + "\n```", "mbpp", test_list=["assert add(1,2)==3"]) == 1.0
    print("code_exec smoke OK")
