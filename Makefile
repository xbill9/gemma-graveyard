.PHONY: test lint

test:
	python3 -m unittest discover -s tests -v

# A NEW TOP-LEVEL MODULE IS UNLINTED UNTIL IT IS ADDED HERE.
lint:
	ruff check graveyard.py f16_range.py lazy_eval.py docs/make_cover.py tests
	bash -n ladder.sh
	@echo "lint OK"
