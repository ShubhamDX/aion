package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ShubhamDX/aion/internal/types"
)

func TestWritePreRequestBlockBudgetExceeded(t *testing.T) {
	w := httptest.NewRecorder()
	writePreRequestBlock(w, types.PreRequestDecision{
		ReasonCode: "budget_exceeded",
		Message:    "The project budget is exhausted.",
	})

	if w.Code != http.StatusPaymentRequired {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusPaymentRequired)
	}
	if got := w.Header().Get("X-AION-Reason-Code"); got != "budget_exceeded" {
		t.Fatalf("X-AION-Reason-Code = %q, want budget_exceeded", got)
	}
	if got := w.Header().Get("Retry-After"); got != "" {
		t.Fatalf("Retry-After = %q, want empty because the hook supplied no reset time", got)
	}
	var body types.ErrorResponse
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Error.Type != "budget_exceeded" || body.Error.Message != "The project budget is exhausted." {
		t.Fatalf("error = %+v", body.Error)
	}
}

func TestWritePreRequestBlockPolicy(t *testing.T) {
	w := httptest.NewRecorder()
	writePreRequestBlock(w, types.PreRequestDecision{ReasonCode: "output_policy_block"})

	if w.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusForbidden)
	}
	if got := w.Header().Get("X-AION-Reason-Code"); got != "output_policy_block" {
		t.Fatalf("X-AION-Reason-Code = %q, want output_policy_block", got)
	}
	var body types.ErrorResponse
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Error.Type != "policy_block" {
		t.Fatalf("error type = %q, want policy_block", body.Error.Type)
	}
}

func TestWritePreRequestBlockLicenseUnavailable(t *testing.T) {
	w := httptest.NewRecorder()
	writePreRequestBlock(w, types.PreRequestDecision{ReasonCode: "license_unavailable"})

	if w.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusForbidden)
	}
	var body types.ErrorResponse
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Error.Type != "license_unavailable" {
		t.Fatalf("error type = %q, want license_unavailable", body.Error.Type)
	}
}
