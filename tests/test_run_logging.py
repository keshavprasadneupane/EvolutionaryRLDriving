import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from evaluation.aggregate import load_run, moving_average
from training.run_logger import EVAL_FIELDS, METRIC_FIELDS, RunLogger


def make_run(root, seed, n_episodes=60, slope=1.0, fail=False):
	rng = np.random.default_rng(seed)
	run = RunLogger("exp", seed=seed, config={"lr": 1e-3, "seed": seed}, root=root, device="cpu")
	try:
		with run:
			for ep in range(n_episodes):
				run.log_episode(
					episode=ep,
					env_steps=(ep + 1) * 100,
					reward=slope * ep + rng.normal(),
					length=100,
					epsilon=max(0.1, 1 - ep / n_episodes),
					loss=None if ep == 0 else 0.5,
				)
			for i in range(3):
				run.log_eval_episode(env_steps=n_episodes * 100, eval_seed=1000 + i, reward=10.0 + i, length=90)
			if fail:
				raise RuntimeError("boom")
	except RuntimeError:
		pass
	return run


def test_run_folder_contents(tmp_path):
	run = make_run(tmp_path, seed=3)
	assert run.dir.parent == tmp_path / "exp" and run.dir.name.endswith("_seed3")
	assert json.loads((run.dir / "config.json").read_text()) == {"lr": 1e-3, "seed": 3}

	info = json.loads((run.dir / "run_info.json").read_text())
	assert info["status"] == "completed" and info["seed"] == 3
	assert info["duration_seconds"] >= 0 and "finished_at" in info
	assert "python" in info["versions"] and info["git_dirty"] in (True, False, None)

	rows = list(csv.reader(open(run.dir / "metrics.csv")))
	assert tuple(rows[0]) == METRIC_FIELDS and len(rows) == 61
	assert rows[1][5] == ""  # missing loss stays blank
	eval_rows = list(csv.reader(open(run.dir / "eval.csv")))
	assert tuple(eval_rows[0]) == EVAL_FIELDS and len(eval_rows) == 4


def test_failed_run_is_marked_failed(tmp_path):
	run = make_run(tmp_path, seed=0, fail=True)
	assert json.loads((run.dir / "run_info.json").read_text())["status"] == "failed"


def test_metrics_are_flushed_before_finish(tmp_path):
	logger = RunLogger("exp", seed=0, config={}, root=tmp_path)
	logger.log_episode(episode=0, env_steps=10, reward=1.0, length=10)
	assert len(list(csv.reader(open(logger.dir / "metrics.csv")))) == 2  # crash-safe
	logger.finish()
	logger.finish()  # second call is harmless


def test_two_runs_same_second_do_not_collide(tmp_path):
	a = RunLogger("exp", seed=0, config={}, root=tmp_path)
	with pytest.raises(FileExistsError):
		RunLogger("exp", seed=0, config={}, root=tmp_path)
	a.finish()


def test_moving_average_and_load_run(tmp_path):
	smooth, used = moving_average(np.arange(10.0), 4)
	assert used == 4 and smooth.tolist() == [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
	assert moving_average(np.arange(3.0), 20)[1] == 3  # window shrinks for short runs

	run = make_run(tmp_path, seed=0, slope=1.0)
	loaded = load_run(run.dir, window=10, target=40.0)
	assert loaded is not None
	assert loaded["episodes"] == 60 and loaded["env_steps"] == 6000
	assert loaded["final_reward"] == pytest.approx(54.5, abs=1.5)  # mean of last 10 of ~0..59
	assert loaded["steps_to_target"] is not None
	assert loaded["eval_mean"] == pytest.approx(11.0)
	assert loaded["eval_std"] == pytest.approx(np.std([10.0, 11.0, 12.0]))
	no_target = load_run(run.dir, window=10, target=1e9)
	assert no_target is not None and no_target["steps_to_target"] is None


def test_aggregate_cli_end_to_end(tmp_path):
	runs_dir, results_dir = tmp_path / "runs", tmp_path / "results"
	for seed in range(3):
		make_run(runs_dir, seed=seed, slope=1.0 + 0.1 * seed)
	make_run(runs_dir, seed=9, fail=True)  # must be excluded by default

	out = subprocess.run(
		[
			sys.executable, "-m", "evaluation.aggregate", "exp",
			"--runs-dir", str(runs_dir), "--results-dir", str(results_dir),
			"--window", "10", "--target", "40",
		],
		capture_output=True, text=True,
	)
	assert out.returncode == 0, out.stderr

	res = results_dir / "exp"
	assert (res / "learning_curve.png").stat().st_size > 1000
	summary = (res / "summary.md").read_text()
	assert "| runs (seeds) | 3 |" in summary
	assert "Excluded" in summary and "_seed9" in summary
	assert "runs reaching target 40 | 3/3" in summary
	assert len(list(csv.reader(open(res / "summary.csv")))) > 5


def test_aggregate_cli_no_runs(tmp_path):
	out = subprocess.run(
		[sys.executable, "-m", "evaluation.aggregate", "nothing", "--runs-dir", str(tmp_path)],
		capture_output=True, text=True,
	)
	assert out.returncode != 0
	assert "not found" in out.stderr or "No usable runs" in out.stderr


def test_aggregate_cli_empty_experiment_folder(tmp_path):
	(tmp_path / "empty").mkdir()
	out = subprocess.run(
		[sys.executable, "-m", "evaluation.aggregate", "empty", "--runs-dir", str(tmp_path)],
		capture_output=True, text=True,
	)
	assert out.returncode != 0 and "No usable runs" in out.stderr


def _git_cmd(cwd, *args):
	subprocess.run(
		["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
		cwd=cwd, check=True, capture_output=True,
	)


_seed = iter(range(1000))


def _dirty(root="runs"):
	# unique seed per call: folder names only have one-second resolution
	run = RunLogger("exp", seed=next(_seed), config={}, root=root)
	info = json.loads((run.dir / "run_info.json").read_text())
	run.finish()
	return info["git_dirty"], info["git_commit"]


def test_git_dirty_ignores_own_outputs_but_catches_code_changes(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	_git_cmd(tmp_path, "init", "-q")
	(tmp_path / "train.py").write_text("print('v1')\n")
	_git_cmd(tmp_path, "add", "-A")
	_git_cmd(tmp_path, "commit", "-qm", "init")

	dirty, commit = _dirty()
	assert dirty is False and commit  # clean tree
	(tmp_path / "results").mkdir()
	(tmp_path / "results" / "x.md").write_text("generated\n")
	dirty, _ = _dirty()  # earlier run folder + results/ exist, still untracked
	assert dirty is False

	(tmp_path / "train.py").write_text("print('v2')\n")  # edited tracked code
	assert _dirty()[0] is True
	_git_cmd(tmp_path, "checkout", "train.py")
	(tmp_path / "new_agent.py").write_text("x = 1\n")  # new untracked code file
	assert _dirty()[0] is True


def test_git_dirty_is_none_outside_a_repo(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
	assert _dirty() == (None, None)