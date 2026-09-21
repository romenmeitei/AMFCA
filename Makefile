.PHONY: test verify plan example

test:
	pytest -q

verify:
	python scripts/verify_repository.py

plan:
	python scripts/run_pipeline.py --config configs/lhende_2026.yaml --plan

example:
	python scripts/run_pipeline.py --config configs/example_synthetic.yaml
