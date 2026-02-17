.PHONY: build test lint run clean docker-build docker-run fmt vet

BINARY_NAME=aion
BUILD_DIR=bin
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS=-ldflags "-s -w -X main.version=$(VERSION)"

build:
	@mkdir -p $(BUILD_DIR)
	go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) ./cmd/aion

run: build
	./$(BUILD_DIR)/$(BINARY_NAME) -config configs/aion.yaml

test:
	go test -race -cover ./...

test-cover:
	go test -race -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

bench:
	go test -bench=. -benchmem ./internal/classifier/

lint: vet
	@which golangci-lint > /dev/null 2>&1 || echo "Install golangci-lint: https://golangci-lint.run/welcome/install/"
	golangci-lint run ./...

vet:
	go vet ./...

fmt:
	gofmt -s -w .

clean:
	rm -rf $(BUILD_DIR) coverage.out coverage.html

docker-build:
	docker build -t aion:$(VERSION) -t aion:latest .

docker-run:
	docker run --rm -p 8080:8080 \
		-v $(PWD)/configs:/app/configs:ro \
		-v $(PWD)/data:/app/data \
		--env-file .env \
		aion:latest
