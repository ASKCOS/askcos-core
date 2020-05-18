################################################################################
#
#   Makefile for ASKCOS
#
################################################################################

.PHONY: build debug push test setup_pages

VERSION ?= dev
GIT_HASH := $(shell git log -1 --format='format:%H')
GIT_DATE := $(shell git log -1 --format='format:%cs')
GIT_DESCRIBE := $(shell git describe --tags --always --dirty)

REGISTRY ?= registry.gitlab.com/mlpds_mit/askcos/askcos
TAG ?= $(VERSION)
DATA_VERSION ?= $(VERSION)

main build:
	@echo Building docker image: $(REGISTRY):$(TAG)
	@sed \
		-e 's/{VERSION}/$(VERSION)/g' \
		-e 's/{GIT_HASH}/$(GIT_HASH)/g' \
		-e 's/{GIT_DATE}/$(GIT_DATE)/g' \
		-e 's/{GIT_DESCRIBE}/$(GIT_DESCRIBE)/g' \
		Dockerfile | docker build -t $(REGISTRY):$(TAG) \
		--build-arg DATA_VERSION=$(DATA_VERSION) \
		-f - .

build_ci:
	@echo Building docker image: $(REGISTRY):$(TAG)
	@sed \
		-e 's/{VERSION}/$(VERSION)/g' \
		-e 's/{GIT_HASH}/$(GIT_HASH)/g' \
		-e 's/{GIT_DATE}/$(GIT_DATE)/g' \
		-e 's/{GIT_DESCRIBE}/$(GIT_DESCRIBE)/g' \
		Dockerfile | docker build -t $(REGISTRY):$(TAG) \
		--cache-from $(REGISTRY):dev \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg DATA_VERSION=$(DATA_VERSION) \
		-f - .

push: build_ci
	@docker push $(REGISTRY):$(TAG)

debug:
	docker run -it --rm -w /usr/local/ASKCOS $(VOLUMES) $(REGISTRY):$(TAG) /bin/bash

test:
	docker run --rm -w /usr/local/ASKCOS $(VOLUMES) $(REGISTRY):$(TAG) /bin/bash -c "python -m unittest discover -v -p '*test.py' -s makeit"

setup_pages:
	@if [ -d "./pages" ]; then echo "directory 'pages' already exists"; exit 1; fi
	git clone --single-branch --branch pages $(shell git remote get-url origin) pages
