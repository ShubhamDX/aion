package proxy

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/ShubhamDX/aion/internal/types"
)

type streamDeliveryWriter struct {
	http.ResponseWriter
	failed bool
}

func (w *streamDeliveryWriter) Write(b []byte) (int, error) {
	n, err := w.ResponseWriter.Write(b)
	if err != nil || n != len(b) {
		w.failed = true
	}
	return n, err
}

// Retain bounded response material in memory only. Never publish partial warmth.
const maxStreamPrefixText = 256 * 1024

type streamPrefix struct {
	text     strings.Builder
	tools    *streamToolBuffer
	finished bool
	invalid  bool
	chunks   int
}

func (p *streamPrefix) add(c *types.ChatCompletionChunk) {
	if p.invalid || c == nil {
		return
	}
	p.chunks++
	if p.chunks > 16384 {
		p.invalidate()
		return
	}
	for _, ch := range c.Choices {
		if ch.Index != 0 || (p.finished && (ch.Delta.Content != nil || len(ch.Delta.ToolCalls) > 0)) {
			p.invalidate()
			return
		}
		if ch.Delta.Content != nil {
			if p.text.Len()+len(*ch.Delta.Content) > maxStreamPrefixText {
				p.invalidate()
				return
			}
			p.text.WriteString(*ch.Delta.Content)
		}
		if len(ch.Delta.ToolCalls) > 0 {
			if p.tools == nil {
				p.tools = newStreamToolBuffer()
			}
			p.tools.add(&types.ChatCompletionChunk{Choices: []types.ChunkChoice{ch}})
			if p.tools.overflow {
				p.invalidate()
				return
			}
		}
		if ch.FinishReason != nil && *ch.FinishReason != "" {
			if *ch.FinishReason != "stop" && *ch.FinishReason != "tool_calls" {
				p.invalidate()
				return
			}
			p.finished = true
		}
	}
}

func (p *streamPrefix) invalidate() { p.invalid = true; p.text.Reset(); p.tools = nil }

func (p *streamPrefix) digest(req *types.ChatCompletionRequest, complete bool) string {
	if !complete || !p.finished || p.invalid {
		return ""
	}
	m := types.Message{Role: "assistant"}
	if p.text.Len() > 0 {
		m.Content, _ = json.Marshal(p.text.String())
	} else {
		m.Content = json.RawMessage(`null`)
	}
	if p.tools != nil {
		calls, err := p.tools.proposed()
		if err != nil {
			return ""
		}
		for _, c := range calls {
			if c.ID == "" || c.Name == "" || !json.Valid([]byte(c.Args)) {
				return ""
			}
			m.ToolCalls = append(m.ToolCalls, types.ToolCall{ID: c.ID, Type: "function", Function: types.FunctionCall{Name: c.Name, Arguments: c.Args}})
		}
	}
	return types.NextCachePrefixMaterial(req, &types.ChatCompletionResponse{Choices: []types.Choice{{Message: m}}})
}
