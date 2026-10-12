IMAGE ?= glm53-flash-2x3090:locked
PYTHON ?= python3

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: validate test release-check build-image preflight serve download-model

validate:
	$(PYTHON) scripts/validate_repo.py

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

release-check: validate test
	$(PYTHON) scripts/check_release_ready.py

build-image: validate
	docker build --pull -f docker/Dockerfile -t $(IMAGE) .

preflight:
	./scripts/preflight.sh

download-model:
	./scripts/download_model.sh "$(MODEL_DIR)"

serve: build-image preflight
	IMAGE=$(IMAGE) ./scripts/docker_serve.sh
