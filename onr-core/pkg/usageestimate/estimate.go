package usageestimate

import (
	"github.com/r9s-ai/open-next-router/onr-core/pkg/dslconfig"
)

const (
	StageUpstream           = "upstream"
	StageEstimateBoth       = "estimate_both"
	StageEstimatePrompt     = "estimate_prompt"
	StageEstimateCompletion = "estimate_completion"
)

const (
	apiChatCompletions = "chat.completions"
	apiEmbeddings      = "embeddings"
	apiResponses       = "responses"
	apiMessages        = "claude.messages"

	apiGeminiGenerateContent       = "gemini.generatecontent"
	apiGeminiStreamGenerateContent = "gemini.streamgeneratecontent"
)

type Input struct {
	API   string
	Model string

	UpstreamUsage *dslconfig.Usage

	// Upstream request/response bodies (JSON for non-stream, SSE for stream).
	RequestBody  []byte
	RequestRoot  map[string]any
	ResponseBody []byte
	StreamTail   []byte
}

type Output struct {
	Usage *dslconfig.Usage
	Stage string
}

type parsedRequestBody struct {
	raw  []byte
	obj  any
	root map[string]any
	err  error
}

type tokenEstimateContext struct {
	text                     string
	completion               bool // is output
	numTools                 int  // number of tool definitions
	numThinkingBlockInput    int
	numThinkingBlockOutput   int
	numFunctionCalls         int
	numFunctionCallOutputs   int
	numCustomToolCalls       int
	numCustomToolCallOutputs int
}

func Estimate(cfg *Config, in Input) Output {
	u, stage := normalizeUpstreamUsage(in.UpstreamUsage)
	// normalizeUpstreamUsage returns one of three states:
	// 1. u == nil, stage == "": no upstream usage object.
	// 2. u != nil, stage == StageUpstream: upstream usage has at least one non-zero signal.
	// 3. u != nil, stage == "": upstream usage exists but is effectively all-zero.
	if cfg == nil || !cfg.IsAPIEnabled(in.API) || !cfg.EstimateWhenMissingOrZero {

		return Output{Usage: u, Stage: stage}
	}

	// State 2 with both scalar token fields present: return upstream usage as-is.
	if u != nil && u.InputTokens > 0 && u.OutputTokens > 0 {
		return Output{Usage: u, Stage: stage}
	}

	var outUsage *dslconfig.Usage
	var estimatePromptSuccessed, estimateCompletionsSuccessed bool
	var outStage string

	if u != nil {
		copied := *u
		outUsage = &copied
	} else {
		outUsage = &dslconfig.Usage{}
	}

	// States 1 and 3 estimate from an empty/all-zero base. State 2 reaches here
	// only when one scalar token field is missing; keep existing upstream fields
	// and estimate only the missing side.
	if outUsage == nil || outUsage.InputTokens == 0 {

		reqParsed := parseRequestBody(in.RequestBody, in.RequestRoot, cfg.MaxRequestBytes)
		reqCtx := extractRequestTextFromParsed(in.API, reqParsed)
		inputTokens := EstimateTokenByModel(in.Model, reqCtx)
		if inputTokens > 0 {
			estimatePromptSuccessed = true
		}
		outUsage.InputTokens = inputTokens

	}

	if outUsage == nil || outUsage.OutputTokens == 0 {
		respText := ""

		if len(in.StreamTail) > 0 {
			respText = extractStreamText(in.API, in.StreamTail, cfg.MaxStreamCollectBytes)
		} else {
			respText = extractResponseTextForModel(in.API, in.Model, in.ResponseBody, cfg.MaxResponseBytes)
		}

		respCtx := &tokenEstimateContext{text: respText, completion: true, numTools: 0}
		outputTokens := EstimateTokenByModel(in.Model, respCtx)
		if outputTokens > 0 {
			estimateCompletionsSuccessed = true
		}
		outUsage.OutputTokens = outputTokens
	}
	if !estimateCompletionsSuccessed && !estimatePromptSuccessed {
		return Output{Usage: u, Stage: stage}
	}

	outUsage.TotalTokens = outUsage.InputTokens + outUsage.OutputTokens
	if estimatePromptSuccessed {
		outStage = StageEstimatePrompt
	}
	if estimateCompletionsSuccessed {
		outStage = StageEstimateCompletion
	}
	if estimatePromptSuccessed && estimateCompletionsSuccessed {
		outStage = StageEstimateBoth
	}

	return Output{Usage: outUsage, Stage: outStage}

}

func normalizeUpstreamUsage(u *dslconfig.Usage) (*dslconfig.Usage, string) {
	if u == nil {
		return nil, ""
	}
	// Copy to avoid mutating callers.
	out := *u

	// Normalize legacy OpenAI fields.
	if out.InputTokens == 0 && out.PromptTokens != 0 {
		out.InputTokens = out.PromptTokens
	}
	if out.OutputTokens == 0 && out.CompletionTokens != 0 {
		out.OutputTokens = out.CompletionTokens
	}
	if out.TotalTokens == 0 && (out.InputTokens != 0 || out.OutputTokens != 0) {
		out.TotalTokens = out.InputTokens + out.OutputTokens
	}

	if isAllZero(&out) {
		return &out, ""
	}
	return &out, StageUpstream
}

func isAllZero(u *dslconfig.Usage) bool {
	if u == nil {
		return true
	}
	return u.InputTokens == 0 && u.OutputTokens == 0 && u.TotalTokens == 0 &&
		(u.InputTokenDetails == nil || (u.InputTokenDetails.CachedTokens == 0 && u.InputTokenDetails.CacheWriteTokens == 0))
}
