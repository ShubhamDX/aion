package types

import (
	"encoding/json"
	"testing"
)

func TestSessionPrefixSurvivesClientUnicodeEncoding(t *testing.T) {
	request := &ChatCompletionRequest{Messages: []Message{{Role: "user", Content: json.RawMessage(`"Review Orion"`)}}}
	response := &ChatCompletionResponse{Choices: []Choice{{Message: Message{Role: "assistant", Content: json.RawMessage(`"Orion — preserve data"`)}}}}
	next := NextCachePrefixMaterial(request, response)
	followup := &ChatCompletionRequest{Messages: []Message{
		request.Messages[0],
		{Role: "assistant", Content: json.RawMessage(`"Orion \u2014 preserve data"`)},
		{Role: "user", Content: json.RawMessage(`"What next?"`)},
	}}
	got := SessionMaterialFromRequest(followup, "same-session").CachePrefixMaterialSHA256
	if got != next || got == "" {
		t.Fatalf("same semantic prefix changed after JSON re-encoding: %s != %s", got, next)
	}
}

func TestSessionPrefixNormalizationPreservesMeaningAndPrecision(t *testing.T) {
	digest := func(raw string) string {
		return messagesDigest([]Message{{Role: "system", Content: json.RawMessage(raw)}})
	}
	if digest(`{"b":"\u2014","a":9007199254740993}`) != digest(`{ "a":9007199254740993, "b":"—" }`) {
		t.Fatal("equivalent encoding differs")
	}
	if digest(`{"a":9007199254740993}`) == digest(`{"a":9007199254740992}`) {
		t.Fatal("numeric precision lost")
	}
	if digest(`"preserve volumes"`) == digest(`"delete volumes"`) {
		t.Fatal("changed content collided")
	}
	if digest(`"a" "b"`) != "" {
		t.Fatal("invalid JSON accepted")
	}
}
