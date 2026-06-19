package provider

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ShubhamDX/aion/pkg/types"
)

func nativeReq(mode string, body *types.JSONSchemaSpec) *types.ChatCompletionRequest {
	r := &types.ChatCompletionRequest{
		Model:    "gpt-4o",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	}
	if mode != "" {
		r.SchemaSettings = &types.SchemaSettings{Mode: mode, SchemaPolicyID: "sp-orders", SchemaBody: body}
	}
	return r
}

func ordersBody() *types.JSONSchemaSpec {
	return &types.JSONSchemaSpec{
		Name:   "orders",
		Strict: true,
		Schema: json.RawMessage(`{"type":"object","required":["id"],"properties":{"id":{"type":"string"}}}`),
	}
}

// SP2b-3b: provider_native + schema body emits OpenAI response_format.json_schema,
// and openAINativePayload reports it emitted.
func TestOpenAINativePayload_EmitsResponseFormat(t *testing.T) {
	payload, emitted := openAINativePayload(nativeReq(types.ProviderSchemaModeProviderNative, ordersBody()), "gpt-4o", false)
	if !emitted {
		t.Fatal("native must be reported emitted")
	}
	if payload.ResponseFormat == nil || payload.ResponseFormat.Type != "json_schema" {
		t.Fatalf("response_format.type must be json_schema, got %+v", payload.ResponseFormat)
	}
	if payload.ResponseFormat.JSONSchema == nil || payload.ResponseFormat.JSONSchema.Name != "orders" || !payload.ResponseFormat.JSONSchema.Strict {
		t.Fatalf("json_schema spec missing: %+v", payload.ResponseFormat.JSONSchema)
	}
	// Golden: exact marshaled shape.
	body, _ := json.Marshal(payload.ResponseFormat)
	want := `{"type":"json_schema","json_schema":{"name":"orders","strict":true,"schema":{"type":"object","required":["id"],"properties":{"id":{"type":"string"}}}}}`
	if string(body) != want {
		t.Fatalf("response_format golden mismatch:\n got:  %s\n want: %s", body, want)
	}
}

// No schema settings, validation_only, or missing body must leave the payload
// unchanged (no response_format added, not reported emitted).
func TestOpenAINativePayload_NonNativeUnchanged(t *testing.T) {
	cases := []struct {
		name string
		req  *types.ChatCompletionRequest
	}{
		{"no settings", nativeReq("", nil)},
		{"validation_only", nativeReq(types.ProviderSchemaModeValidationOnly, ordersBody())},
		{"native missing body", nativeReq(types.ProviderSchemaModeProviderNative, nil)},
	}
	bare, _ := json.Marshal(upstreamPayload(nativeReq("", nil), "gpt-4o", false))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			payload, emitted := openAINativePayload(tc.req, "gpt-4o", false)
			if emitted {
				t.Fatal("must not report native emitted")
			}
			if payload.ResponseFormat != nil {
				t.Fatalf("must not add response_format, got %+v", payload.ResponseFormat)
			}
			got, _ := json.Marshal(payload)
			if string(got) != string(bare) {
				t.Fatalf("payload changed:\n got:  %s\n bare: %s", got, bare)
			}
		})
	}
}

// A caller-supplied response_format must NOT be clobbered by native emission.
func TestOpenAINativePayload_DoesNotClobberCallerFormat(t *testing.T) {
	req := nativeReq(types.ProviderSchemaModeProviderNative, ordersBody())
	req.ResponseFormat = &types.ResponseFormat{Type: "json_object"}
	payload, emitted := openAINativePayload(req, "gpt-4o", false)
	if emitted {
		t.Fatal("must not emit native over a caller-supplied response_format")
	}
	if payload.ResponseFormat.Type != "json_object" {
		t.Fatalf("caller response_format must survive, got %+v", payload.ResponseFormat)
	}
}

// The raw schema body must never appear on a NON-OpenAI path. Anthropic
// translateRequest builds its own struct and must not carry it.
func TestAnthropicTranslate_SchemaBodyNeverLeaks(t *testing.T) {
	p := &AnthropicProvider{}
	req := nativeReq(types.ProviderSchemaModeProviderNative, ordersBody())
	req.Model = "claude-x"
	body, err := json.Marshal(p.translateRequest(req, "claude-3", false))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, banned := range []string{"json_schema", "response_format", "sp-orders", "\"orders\"", "properties"} {
		if strings.Contains(string(body), banned) {
			t.Fatalf("schema body %q leaked into Anthropic payload: %s", banned, body)
		}
	}
}

// Direct request marshal must never emit the carrier or schema body.
func TestChatCompletionRequest_SchemaBodyJSONIgnored(t *testing.T) {
	req := nativeReq(types.ProviderSchemaModeProviderNative, ordersBody())
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, banned := range []string{"SchemaSettings", "schema_settings", "SchemaBody", "sp-orders", "\"orders\""} {
		if strings.Contains(string(body), banned) {
			t.Fatalf("carrier/body %q must never serialize on the request, got: %s", banned, body)
		}
	}
}
