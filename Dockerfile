FROM golang:1.23-alpine AS builder

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o aion ./cmd/aion

FROM alpine:3.19

RUN apk --no-cache add ca-certificates tzdata && \
    adduser -D -H -h /app aion

WORKDIR /app

COPY --from=builder /build/aion .
COPY --from=builder /build/migrations ./migrations

RUN mkdir -p /app/data && chown -R aion:aion /app

USER aion

EXPOSE 8080

ENTRYPOINT ["./aion"]
CMD ["-config", "configs/aion.yaml"]
