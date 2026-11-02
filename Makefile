.PHONY: test lint

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS UNLINTED UNTIL IT IS ADDED HERE.
lint:
	ruff check graveyard.py f16_range.py lazy_eval.py docs/make_cover.py docs/think_derived.py docs/tradeoff.py docs/gpu_derived.py docs/make_cover_gpu.py docs/gpu3_derived.py docs/gpu4_derived.py docs/gpu4_arith.py docs/make_cover_gpu4.py tools/replay_input.py tools/fork_input.py tools/fork_gaps.py tools/branch_test.py tools/flip_classes.py tools/flip_core.py tools/flip_kinds.py tools/flip_arith.py tools/flip_modes.py tools/fork_probe.py tests
	bash -n ladder.sh
	@echo "lint OK"
