-- AION telemetry schema

CREATE TABLE IF NOT EXISTS request_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    request_id  TEXT    NOT NULL,
    api_key_id  TEXT    NOT NULL,
    tier        INTEGER NOT NULL,
    model       TEXT    NOT NULL,
    provider    TEXT    NOT NULL,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL    NOT NULL DEFAULT 0.0,
    savings_usd     REAL    NOT NULL DEFAULT 0.0,
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    status_code     INTEGER NOT NULL DEFAULT 200,
    stream          BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_request_log_created_at ON request_log(created_at);
CREATE INDEX IF NOT EXISTS idx_request_log_api_key_id ON request_log(api_key_id);
CREATE INDEX IF NOT EXISTS idx_request_log_model ON request_log(model);

CREATE TABLE IF NOT EXISTS budget_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key_id  TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    cost_usd    REAL    NOT NULL DEFAULT 0.0,
    UNIQUE(api_key_id, date)
);

CREATE INDEX IF NOT EXISTS idx_budget_usage_key_date ON budget_usage(api_key_id, date);
