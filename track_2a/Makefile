.PHONY: run build

IMAGE ?= hackapertus-ost
INPUT ?= data/contract_test.jsonl
OUTPUT ?= data/predictions.jsonl
ENV_FILE_FLAG = $(if $(wildcard .env),--env-file .env,)

build:
	docker build -t $(IMAGE) .

run: build
	docker run --rm $(ENV_FILE_FLAG) \
		-v "$(CURDIR)/data:/app/data" \
		$(IMAGE) \
		--input "/app/$(INPUT)" \
		--output "/app/$(OUTPUT)"
