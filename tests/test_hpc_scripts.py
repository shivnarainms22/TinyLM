"""Static checks for HPC submission scripts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HPC_JOB = ROOT / "scripts" / "hpc_job.sh"
SUBMIT_HPC = ROOT / "scripts" / "submit_hpc.sh"
TOKENIZE_V2_E1 = ROOT / "scripts" / "tokenize_v2_e1_job.sh"
BUILD_V2_E2 = ROOT / "scripts" / "tokenize_v2_e2_job.sh"


def _read(path: Path) -> str:
    assert path.exists(), f"missing: {path}"
    return path.read_text(encoding="utf-8")


def test_hpc_job_accepts_phase_specific_init_and_shards():
    body = _read(HPC_JOB)
    assert 'export TINYLM_SHARD_DIR="${SHARD_DIR:-${SCRATCH}/tinylm/data}"' in body
    assert "TINYLM_INIT_FROM" in body
    assert "INIT_FROM" in body


def test_hpc_job_entry_module_is_configurable_and_defaults_to_train():
    """The v4 KD probe reuses this rechain/SIGTERM job via TINYLM_MODULE; the
    default must stay tinylm.train so every existing run is unchanged."""
    body = _read(HPC_JOB)
    assert 'python -m "${TINYLM_MODULE:-tinylm.train}"' in body


def test_hpc_job_guards_the_scratch_purge_codec_failure():
    assert "unset PYTHONHOME PYTHONPATH" in _read(HPC_JOB)


def test_submit_kd_points_the_shared_job_at_the_kd_module():
    body = _read(ROOT / "scripts" / "submit_kd.sh")
    assert 'export TINYLM_MODULE="tinylm.kd"' in body
    assert "submit_hpc.sh" in body
    assert "configs/v4/run_KD_tinyllama_fwe.yaml" in body
    assert "2000" in body                              # matches E1's 2.1B budget


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


def test_e2_build_job_runs_mixture_builder():
    """E2 builds the broader mixture (2.1B) via the proportional builder."""
    body = _read(BUILD_V2_E2)
    assert "build_mixture_shards.py" in body
    assert "mixed_web_code_math" in body
    assert "--max-shards 21" in body


# ---------------------------------------------------------------------------
# v3 job scripts
#
# sbatch exports the submitting shell's environment by default. Submitting from
# a shell with another project's PYTHONHOME/PYTHONPATH set made the tinylm
# interpreter unable to locate its own stdlib:
#   LookupError: no codec search functions registered: can't find encoding
# Every eval in the job died on startup, and because each was guarded with
# `|| echo WARNING ... continuing`, the job still exited 0 and SLURM reported
# success while writing no results at all.
# ---------------------------------------------------------------------------

V3_JOBS = [
    ROOT / "scripts" / "eval_v3_fewshot_job.sh",
    ROOT / "scripts" / "eval_v3_ppl_job.sh",
    ROOT / "scripts" / "eval_v3_sft_job.sh",
    ROOT / "scripts" / "sft_smoltalk_job.sh",
]


def test_v3_jobs_sanitize_inherited_interpreter_env():
    """Each v3 job must clear PYTHONHOME/PYTHONPATH inherited from the
    submitting shell before running python."""
    for job in V3_JOBS:
        body = _read(job)
        assert "unset PYTHONHOME PYTHONPATH" in body, f"{job.name} does not sanitize the env"
        assert body.index("unset PYTHONHOME PYTHONPATH") < body.index("python "), (
            f"{job.name} sanitizes the env after invoking python"
        )


def test_setup_hpc_rebuilds_a_broken_env_instead_of_skipping_it():
    """`conda env list` showing 'tinylm' does not mean the env works: ~/.conda
    lives on /scratch, and a purge reaped the stdlib .py sources (leaving
    __pycache__), so python died at startup. Setup must verify the interpreter
    runs and recreate the env if it does not."""
    body = _read(ROOT / "scripts" / "setup_hpc.sh")
    assert "conda run -n tinylm python" in body, "setup does not health-check the interpreter"
    assert "conda env remove -n tinylm" in body, "setup cannot recreate a broken env"


def test_v3_multi_eval_jobs_fail_loudly_when_every_eval_fails():
    """The per-eval `|| ... continuing` guard must not let a job that produced
    nothing exit 0 — the job has to propagate a failure to SLURM."""
    for job in (ROOT / "scripts" / "eval_v3_fewshot_job.sh",
                ROOT / "scripts" / "eval_v3_ppl_job.sh"):
        body = _read(job)
        assert "FAILURES=" in body, f"{job.name} does not track failures"
        assert "exit 1" in body, f"{job.name} cannot report failure to SLURM"
