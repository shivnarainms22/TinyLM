"""Static checks for HPC submission scripts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HPC_JOB = ROOT / "scripts" / "hpc_job.sh"
SUBMIT_HPC = ROOT / "scripts" / "submit_hpc.sh"
TOKENIZE_V2_E1 = ROOT / "scripts" / "tokenize_v2_e1_job.sh"


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


def test_e1_tokenize_job_is_disjoint_from_run_d():
    """E1 must skip exactly Run D's 8B sample-100BT prefix and take 21 shards
    (2.1B). These constants encode the non-overlap guarantee — guard against
    silent drift that would re-tokenize Run D's data."""
    body = _read(TOKENIZE_V2_E1)
    assert "--split sample-100BT" in body          # same stream as Run D
    assert "--skip-tokens 8000000000" in body      # step past Run D's 8B prefix
    assert "--max-shards 21" in body               # 2.1B fresh tokens
