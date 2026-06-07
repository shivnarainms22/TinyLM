"""Static checks for HPC submission scripts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HPC_JOB = ROOT / "scripts" / "hpc_job.sh"
SUBMIT_HPC = ROOT / "scripts" / "submit_hpc.sh"


def _read(path: Path) -> str:
    assert path.exists(), f"missing: {path}"
    return path.read_text(encoding="utf-8")


def test_hpc_job_accepts_phase_specific_init_and_shards():
    body = _read(HPC_JOB)
    assert 'export TINYLM_SHARD_DIR="${SHARD_DIR:-${SCRATCH}/tinylm/data}"' in body
    assert "TINYLM_INIT_FROM" in body
    assert "INIT_FROM" in body


def test_submit_hpc_passes_optional_phase_env():
    body = _read(SUBMIT_HPC)
    assert 'INIT_FROM="${4:-}"' in body
    assert 'SHARD_DIR="${5:-}"' in body
    assert 'INIT_FROM="${INIT_FROM}"' in body
    assert 'SHARD_DIR="${SHARD_DIR}"' in body
