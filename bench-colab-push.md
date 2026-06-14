# `make bench-colab-push`

This document explains how to add a `make bench-colab-push` command to a Python package repository so that Google Colab can always pull benchmark code from a dedicated Git branch named `bench-colab`.

## Goal

When developing a Python package locally, the benchmark workflow is usually:

1. Edit the package locally in VS Code.
2. Commit the changes you want to test.
3. Run `make bench-colab-push`.
4. Open a Colab notebook connected to a GPU runtime.
5. In Colab, pull or clone the `bench-colab` branch.
6. Install the package.
7. Run the benchmark.
8. Save results to Google Drive so they sync back to your computer.

The `bench-colab` branch is not meant to be your main development branch. It is a moving branch used only as a stable target for Colab.

## Why use a dedicated branch?

Colab runs on a remote machine. It cannot directly see your local VS Code workspace unless you use a synchronization mechanism.

Using Git is the simplest and most reproducible solution:

- your local code is pushed to GitHub, GitLab, or another remote;
- Colab clones a known branch;
- the branch name never changes;
- the Colab notebook does not need to know your current feature branch name.

Instead of editing the Colab notebook every time you switch local branches, the notebook can always use:

```bash
git clone --branch bench-colab <repo-url> /content/pkg
```

## Makefile target

Add this to your `Makefile`:

```makefile
BENCH_COLAB_BRANCH ?= bench-colab
GIT_REMOTE ?= origin

.PHONY: bench-colab-push
bench-colab-push:
	@set -eu; \
	current_branch=$$(git rev-parse --abbrev-ref HEAD); \
	current_sha=$$(git rev-parse HEAD); \
	if [ -n "$$(git status --porcelain)" ]; then \
		echo "ERROR: working tree is not clean."; \
		echo ""; \
		git status --short; \
		echo ""; \
		echo "Commit or stash your changes before pushing to Colab."; \
		echo "Recommended:"; \
		echo "  git add -A"; \
		echo "  git commit -m 'Benchmark Colab run'"; \
		echo "  make bench-colab-push"; \
		exit 1; \
	fi; \
	echo "Current branch: $$current_branch"; \
	echo "Current commit: $$current_sha"; \
	if [ "$$current_branch" = "$(BENCH_COLAB_BRANCH)" ]; then \
		echo "Already on $(BENCH_COLAB_BRANCH)."; \
	else \
		echo "Updating local branch $(BENCH_COLAB_BRANCH) -> $$current_sha"; \
		git branch -f "$(BENCH_COLAB_BRANCH)" "$$current_sha"; \
	fi; \
	echo "Pushing $(BENCH_COLAB_BRANCH) to $(GIT_REMOTE)..."; \
	git push --force-with-lease "$(GIT_REMOTE)" "$(BENCH_COLAB_BRANCH):$(BENCH_COLAB_BRANCH)"; \
	echo ""; \
	echo "Done."; \
	echo "Colab can now use branch: $(BENCH_COLAB_BRANCH)"; \
	echo "Commit pushed: $$current_sha"
```

Important: Makefile commands must start with a real tab character, not spaces.

## What the target does

The target performs these steps:

1. Reads the current Git branch.
2. Reads the current commit SHA.
3. Checks that the working tree is clean.
4. Creates or updates a local branch named `bench-colab` pointing to the current commit.
5. Pushes that branch to the remote using `--force-with-lease`.

The command does not switch your current branch. If you are working on `feature/my-benchmark`, you will remain on `feature/my-benchmark` after the command finishes.

## Why require a clean working tree?

Colab pulls from Git. Uncommitted local changes are not available to Colab.

For that reason, this target refuses to run if you have:

- unstaged changes;
- staged but uncommitted changes;
- untracked files.

This avoids the common mistake where Colab runs old code because your newest local changes were never committed.

## How to use it locally

From your local repository:

```bash
git add -A
git commit -m "Benchmark Colab run"
make bench-colab-push
```

You should see output similar to:

```text
Current branch: my-feature-branch
Current commit: abc1234...
Updating local branch bench-colab -> abc1234...
Pushing bench-colab to origin...
Done.
Colab can now use branch: bench-colab
Commit pushed: abc1234...
```

## Using a different remote or branch name

The defaults are:

```makefile
BENCH_COLAB_BRANCH ?= bench-colab
GIT_REMOTE ?= origin
```

You can override them from the command line:

```bash
make bench-colab-push BENCH_COLAB_BRANCH=bench-gpu GIT_REMOTE=upstream
```

For the standard Colab workflow, keep the default branch name:

```bash
make bench-colab-push
```

## Colab runner example

In your Colab notebook, use the dedicated branch:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then run:

```python
import datetime
import pathlib
import subprocess

REPO_URL = "https://github.com/YOUR_USER/YOUR_REPO.git"
BRANCH = "bench-colab"
PKG_DIR = "/content/pkg"

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = pathlib.Path(f"/content/drive/MyDrive/colab_benchmarks/{run_id}")
out_dir.mkdir(parents=True, exist_ok=True)

def sh(command, cwd=None):
    print(f"\n$ {command}")
    subprocess.run(command, shell=True, cwd=cwd, check=True)

sh(f"rm -rf {PKG_DIR}")
sh(f"git clone --depth 1 --branch {BRANCH} {REPO_URL} {PKG_DIR}")
sh("python -m pip install -U pip", cwd=PKG_DIR)
sh('python -m pip install -e ".[bench]"', cwd=PKG_DIR)
sh("nvidia-smi || true")
sh(f"python -m your_package.benchmark --out {out_dir}", cwd=PKG_DIR)

print(f"Benchmark results written to: {out_dir}")
```

Replace:

- `YOUR_USER/YOUR_REPO` with your repository path;
- `your_package.benchmark` with your actual benchmark module;
- `.[bench]` with your real package extra, if different.

## Result synchronization

The benchmark output directory is inside Google Drive:

```text
/content/drive/MyDrive/colab_benchmarks/<run_id>
```

If you use Google Drive for desktop on your computer, this directory can automatically sync back to your local machine.

## Troubleshooting

### `make: *** missing separator`

Your Makefile probably uses spaces instead of tabs. Every command line inside a Makefile target must start with a tab.

### `ERROR: working tree is not clean`

You have local changes that are not committed. Run:

```bash
git status
```

Then either commit them:

```bash
git add -A
git commit -m "Benchmark Colab run"
make bench-colab-push
```

or stash them:

```bash
git stash --include-untracked
make bench-colab-push
```

### Colab still runs old code

Common causes:

1. You forgot to commit your changes before running `make bench-colab-push`.
2. The notebook clones another branch instead of `bench-colab`.
3. The package was already installed in the Colab runtime.

A safe reset in Colab is:

```bash
rm -rf /content/pkg
python -m pip uninstall -y your-package-name
```

Then clone and install again.

### Push is rejected

The target uses `--force-with-lease`, which is safer than `--force`. It refuses to overwrite remote changes that you do not have locally.

If this happens, inspect the remote branch before forcing anything:

```bash
git fetch origin bench-colab
git log --oneline --decorate --graph origin/bench-colab -n 10
```

## Recommended workflow

Use this loop while developing benchmarks:

```bash
# local machine
git add -A
git commit -m "Benchmark experiment"
make bench-colab-push
```

Then in Colab:

```bash
git clone --depth 1 --branch bench-colab <repo-url> /content/pkg
cd /content/pkg
python -m pip install -e ".[bench]"
python -m your_package.benchmark --out /content/drive/MyDrive/colab_benchmarks/<run_id>
```

This gives you a clean split:

- local machine: development and Git push;
- Colab: GPU execution;
- Google Drive: result synchronization.
