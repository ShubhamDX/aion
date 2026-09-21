package provider

import (
	"context"
	"encoding/json"
	"github.com/ShubhamDX/aion/internal/config"
	"github.com/ShubhamDX/aion/internal/types"
	"io"
	"os"
	"testing"
	"time"
)

// Opt-in only. Reuses the supplied credential and makes one bounded paid call.
func TestNovaLiveStream(t *testing.T) {
	if os.Getenv("AION_RUN_NOVA_LIVE") != "1" {
		t.Skip("set AION_RUN_NOVA_LIVE=1 to make a paid Bedrock call")
	}
	token := os.Getenv("AWS_BEARER_TOKEN_BEDROCK")
	if token == "" {
		t.Fatal("existing Bedrock credential required")
	}
	p, err := NewBedrock(&config.ProviderConfig{APIKey: token, Region: os.Getenv("AWS_REGION")})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	limit := 100
	stream, err := p.SendStream(ctx, &types.ChatCompletionRequest{Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"Return the word READY."`)}}, MaxTokens: &limit}, "us.amazon.nova-2-lite-v1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	chunks := 0
	usage := false
	for {
		chunk, err := stream.ReadChunk()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		chunks++
		if chunk.Usage != nil {
			usage = chunk.Usage.TotalTokens > 0 && chunk.Usage.InputPartitionValid()
		}
		raw, _ := json.Marshal(chunk)
		t.Log(string(raw))
	}
	if chunks == 0 || !usage {
		t.Fatal("stream lacked chunks or consistent usage")
	}
}
