package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestCountTokens replaces the old always-zero stub's contract. It doesn't
// have access to a real provider tokenizer, so this asserts a defined
// heuristic estimate is returned (non-zero for non-empty content, scaling
// with input length) and that the response explicitly flags itself as an
// estimate rather than an exact count.
func TestCountTokens(t *testing.T) {
	h := &Handler{}

	t.Run("short prompt returns a positive estimate flagged as an estimate", func(t *testing.T) {
		body := `{"model":"claude-haiku","messages":[{"role":"user","content":"The quick brown fox jumps over the lazy dog."}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(body))
		rec := httptest.NewRecorder()

		h.CountTokens(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
		}
		var resp struct {
			InputTokens int  `json:"input_tokens"`
			Estimate    bool `json:"estimate"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode response: %v; body=%s", err, rec.Body.String())
		}
		if resp.InputTokens <= 0 {
			t.Fatalf("input_tokens = %d, want > 0 for a non-empty prompt (this is the exact bug being fixed: always returning 0)", resp.InputTokens)
		}
		if !resp.Estimate {
			t.Fatalf("estimate = false, want true — AION has no exact provider tokenizer here and must say so")
		}
	})

	t.Run("longer prompt yields a larger estimate than a shorter one", func(t *testing.T) {
		short := `{"model":"claude-haiku","messages":[{"role":"user","content":"hi"}]}`
		long := `{"model":"claude-haiku","messages":[{"role":"user","content":"` + strings.Repeat("word ", 500) + `"}]}`

		shortRec := httptest.NewRecorder()
		h.CountTokens(shortRec, httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(short)))
		longRec := httptest.NewRecorder()
		h.CountTokens(longRec, httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(long)))

		var shortResp, longResp struct {
			InputTokens int `json:"input_tokens"`
		}
		_ = json.Unmarshal(shortRec.Body.Bytes(), &shortResp)
		_ = json.Unmarshal(longRec.Body.Bytes(), &longResp)

		if longResp.InputTokens <= shortResp.InputTokens {
			t.Fatalf("long prompt estimate (%d) should exceed short prompt estimate (%d)", longResp.InputTokens, shortResp.InputTokens)
		}
	})

	// The Claude Code preflight this endpoint exists to serve is dominated by
	// the system prompt and the tool schemas rather than by the message text,
	// so a count that ignored either would be useless in practice.
	t.Run("system prompt is counted", func(t *testing.T) {
		const systemChars = 2500 // 500 repetitions of "word "
		base := countTokensEstimate(t, h, `{"model":"claude-haiku","messages":[{"role":"user","content":"hi"}]}`)
		withSystem := countTokensEstimate(t, h, `{"model":"claude-haiku","system":"`+strings.Repeat("word ", 500)+`","messages":[{"role":"user","content":"hi"}]}`)

		if want := base + systemChars/4 - 100; withSystem < want {
			t.Fatalf("a %d-char system prompt moved the estimate from %d to %d, want at least %d", systemChars, base, withSystem, want)
		}
	})

	t.Run("tool definitions are counted", func(t *testing.T) {
		const schemaChars = 2000
		base := countTokensEstimate(t, h, `{"model":"claude-haiku","messages":[{"role":"user","content":"hi"}]}`)
		withTools := countTokensEstimate(t, h, `{"model":"claude-haiku","messages":[{"role":"user","content":"hi"}],"tools":[{"name":"f","description":"d","input_schema":{"type":"object","properties":{"p":{"type":"string","description":"`+strings.Repeat("x", schemaChars)+`"}}}}]}`)

		if want := base + schemaChars/4 - 100; withTools < want {
			t.Fatalf("a %d-char tool schema moved the estimate from %d to %d, want at least %d", schemaChars, base, withTools, want)
		}
	})

	// A tool_result turn can carry trailing text alongside the results (e.g.
	// "given the result above, do X"). The translator used to recognize the
	// first block as tool_result and only ever emit tool_result blocks,
	// silently dropping any trailing text block from that same message —
	// not just from the token estimate, but from the request sent to the
	// provider entirely.
	t.Run("trailing text after a tool_result block is counted, not dropped", func(t *testing.T) {
		const textChars = 100_000
		withoutText := `{"model":"claude-haiku","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}]}`
		withText := `{"model":"claude-haiku","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"},{"type":"text","text":"` + strings.Repeat("x", textChars) + `"}]}]}`

		base := countTokensEstimate(t, h, withoutText)
		withTrailingText := countTokensEstimate(t, h, withText)

		if want := base + textChars/4 - 100; withTrailingText < want {
			t.Fatalf("a %d-char trailing text block moved the estimate from %d to %d, want at least %d", textChars, base, withTrailingText, want)
		}
	})

	t.Run("empty messages is rejected, not silently estimated as zero", func(t *testing.T) {
		body := `{"model":"claude-haiku","messages":[]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(body))
		rec := httptest.NewRecorder()

		h.CountTokens(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400 for empty messages; body=%s", rec.Code, rec.Body.String())
		}
	})

	t.Run("malformed JSON body", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(`{`))
		rec := httptest.NewRecorder()

		h.CountTokens(rec, req)

		if rec.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400 for malformed JSON; body=%s", rec.Code, rec.Body.String())
		}
	})
}

func countTokensEstimate(t *testing.T, h *Handler, body string) int {
	t.Helper()
	rec := httptest.NewRecorder()
	h.CountTokens(rec, httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var resp struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v; body=%s", err, rec.Body.String())
	}
	return resp.InputTokens
}
