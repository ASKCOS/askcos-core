################################################################################
#
#   Makefile for ASKCOS
#
################################################################################

.PHONY: build debug push test

VERSION ?= dev
GIT_HASH := $(shell git log -1 --format='format:%H')
GIT_DATE := $(shell git log -1 --format='format:%cs')
GIT_DESCRIBE := $(shell git describe --tags --always --dirty)

REGISTRY ?= registry.gitlab.com/mlpds_mit/askcos/askcos
TAG ?= $(VERSION)

main build:
	@echo Building docker image: $(REGISTRY):$(TAG)
	@sed \
		-e 's/{VERSION}/$(VERSION)/g' \
		-e 's/{GIT_HASH}/$(GIT_HASH)/g' \
		-e 's/{GIT_DATE}/$(GIT_DATE)/g' \
		-e 's/{GIT_DESCRIBE}/$(GIT_DESCRIBE)/g' \
		Dockerfile | docker build -t $(REGISTRY):$(TAG) -f - .

push: build
	@docker push $(REGISTRY):$(TAG)

debug:
	docker run -it -w /usr/local/ASKCOS $(VOLUMES) $(REGISTRY):$(TAG) /bin/bash

test:
	docker run -w /usr/local/ASKCOS $(VOLUMES) $(REGISTRY):$(TAG) /bin/bash -c "python -m unittest discover -v -p '*test.py' -s makeit"
