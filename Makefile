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
		echo "Commit or stash your changes before pushing a Colab benchmark branch."; \
		echo "Recommended:"; \
		echo "  git add -A"; \
		echo "  git commit -m 'Benchmark Colab run'"; \
		echo "  make bench-colab-push"; \
		exit 1; \
	fi; \
	echo "Current branch: $$current_branch"; \
	echo "Current commit: $$current_sha"; \
	if [ "$$current_branch" != "$(BENCH_COLAB_BRANCH)" ]; then \
		echo "Updating local branch $(BENCH_COLAB_BRANCH) -> $$current_sha"; \
		git branch -f "$(BENCH_COLAB_BRANCH)" "$$current_sha"; \
	else \
		echo "Already on $(BENCH_COLAB_BRANCH)."; \
	fi; \
	echo "Pushing $(BENCH_COLAB_BRANCH) to $(GIT_REMOTE)..."; \
	git push --force-with-lease "$(GIT_REMOTE)" "$(BENCH_COLAB_BRANCH):$(BENCH_COLAB_BRANCH)"; \
	echo ""; \
	echo "Done."; \
	echo "Colab can now clone branch: $(BENCH_COLAB_BRANCH)"; \
	echo "Commit pushed: $$current_sha"
