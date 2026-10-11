ENV_NAME := swe-airline-delay

.PHONY: setup test lint format smoke clean-data
setup:  ## create the conda env and install the package (editable)
	conda env create -f environment.yml || conda env update -f environment.yml --prune
	conda run -n $(ENV_NAME) pip install -e . --no-deps

test:
	pytest

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

smoke:  ## run the end-to-end notebook headlessly (needs network unless OFFLINE_DEMO=1)
	jupyter nbconvert --to notebook --execute notebooks/00_pipeline_smoke_test.ipynb --output /tmp/smoke_out.ipynb

clean-data:  ## remove interim + processed data (raw downloads are kept)
	find data/interim data/processed -mindepth 1 ! -name .gitkeep -delete
