"""Static checks for scripts/run_training.sh.

We can't exercise the live tmux launch on Windows CI, so we verify the
script's shape: shebang, safety flags, required commands, error paths.
This is intentionally a structural / lint test, not a behavioral one.
"""

from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_training.sh"


def _read() -> str:
    assert SCRIPT.exists(), f"missing: {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


def test_script_exists():
    assert SCRIPT.is_file()


def test_has_bash_shebang():
    first_line = _read().splitlines()[0]
    assert first_line.startswith("#!"), f"missing shebang: {first_line!r}"
    assert "bash" in first_line, f"not a bash script: {first_line!r}"


def test_uses_strict_mode():
    """set -euo pipefail catches missing args, unset vars, broken pipes."""
    body = _read()
    assert "set -euo pipefail" in body, (
        "script must use 'set -euo pipefail' for safe failure handling"
    )


def test_validates_config_arg():
    """Script must reject missing/nonexistent config before launching tmux."""
    body = _read()
    assert "if [ $# -lt 1 ]" in body, "must check that a config arg was passed"
    assert "if [ ! -f \"$CONFIG\" ]" in body, (
        "must verify config file exists before launching"
    )


def test_checks_tmux_installed():
    """If tmux isn't installed, the script should say so, not crash silently."""
    body = _read()
    assert "command -v tmux" in body
    assert "apt-get install" in body, (
        "should hint how to install tmux when missing"
    )


def test_refuses_to_overwrite_existing_session():
    """Re-running with the same session name must NOT double-launch training."""
    body = _read()
    assert "tmux has-session" in body
    assert "already exists" in body


def test_invokes_training_module():
    """Must run `python -m tinylm.train` with the config arg."""
    body = _read()
    assert "python -u -m tinylm.train" in body, (
        "must use unbuffered python (-u) for real-time log streaming"
    )
    assert "'$CONFIG'" in body, "must pass the validated config to the trainer"


def test_writes_timestamped_log():
    """Output must be tee'd to a timestamped file under logs/."""
    body = _read()
    assert "mkdir -p logs" in body
    assert "tee" in body
    assert "$(date" in body, "log filename must include a timestamp"


def test_prints_reattach_instructions():
    """User-facing output must explain how to attach, detach, and kill."""
    body = _read()
    assert "tmux attach" in body
    assert "Ctrl+B" in body
    assert "kill-session" in body
