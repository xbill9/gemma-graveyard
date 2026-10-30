.PHONY: test lint

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS UNLINTED UNTIL IT IS ADDED HERE.
lint:
	ruff check graveyard.py f16_range.py lazy_eval.py docs/make_cover.py docs/think_derived.py docs/tradeoff.py docs/gpu_derived.py docs/make_cover_gpu.py docs/gpu3_derived.py docs/gpu4_derived.py docs/gpu4_arith.py docs/make_cover_gpu4.py tests
	bash -n ladder.sh
	@echo "lint OK"
