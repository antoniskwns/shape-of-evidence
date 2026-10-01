.DEFAULT_GOAL := paper

PYTHON ?= python3
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all tape figures paper clean

all: paper

# Rebuild the price series from the order book. Needs pandas and pyarrow, and is
# not part of a normal build, since the CSV is included.
tape:
	$(PYTHON) code/extract_market_tape.py

figures:
	$(PYTHON) code/decision_pipeline.py

paper: figures
	cd paper && $(LATEXMK) -jobname='The Shape of Evidence' main.tex

# Build artefacts only. The submitted PDF is a tracked output, so this uses
# latexmk -c rather than -C, which would delete it.
clean:
	cd paper && latexmk -c -jobname='The Shape of Evidence' main.tex
	rm -rf code/__pycache__
	find . -name .DS_Store -delete
