package config

import (
	"fmt"
	"net/url"
	"regexp"
	"strings"
)

// ValidateCloudCredentials validates opt-in cloud identity without discovering
// credentials or contacting an identity service.
func ValidateCloudCredentials(name string, cfg *ProviderConfig) error {
	mode := cfg.CredentialMode
	if mode == "" || mode == "api_key" || mode == "bearer" {
		return nil
	}
	google := mode == "google_adc" && (name == "vertex" || name == "gemini")
	azure := mode == "azure_entra" && name == "openai"
	if !google && !azure {
		return fmt.Errorf("%s: unsupported credential_mode", name)
	}
	if cfg.APIKey != "" {
		return fmt.Errorf("%s: managed credential mode cannot also set api_key", name)
	}
	if name == "vertex" {
		if cfg.ProjectID == "" || strings.ContainsAny(cfg.ProjectID, "/?#\\") {
			return fmt.Errorf("%s: managed credentials require an inference project_id", name)
		}
		if cfg.Region != "" && !regexp.MustCompile(`^[a-z0-9]+(?:-[a-z0-9]+)*$`).MatchString(cfg.Region) {
			return fmt.Errorf("%s: invalid cloud region", name)
		}
		if cfg.BaseURL == "" {
			return nil
		}
	}
	u, err := url.Parse(cfg.BaseURL)
	if err != nil || u.Scheme != "https" || u.User != nil || u.RawQuery != "" ||
		u.Fragment != "" || (u.Port() != "" && u.Port() != "443") {
		return fmt.Errorf("%s: managed credentials require a canonical HTTPS cloud endpoint", name)
	}
	host := strings.ToLower(u.Hostname())
	if google && (host == "aiplatform.googleapis.com" || strings.HasSuffix(host, "-aiplatform.googleapis.com")) {
		if name == "vertex" && strings.Trim(u.Path, "/") == "" {
			return nil
		}
		parts := strings.Split(strings.Trim(u.Path, "/"), "/")
		if name == "gemini" && len(parts) == 7 && parts[0] == "v1" && parts[1] == "projects" &&
			parts[2] != "" && parts[3] == "locations" && parts[4] != "" &&
			parts[5] == "endpoints" && parts[6] == "openapi" {
			return nil
		}
	}
	if azure && (strings.HasSuffix(host, ".openai.azure.com") || strings.HasSuffix(host, ".services.ai.azure.com")) &&
		strings.TrimRight(u.Path, "/") == "/openai/v1" {
		return nil
	}
	return fmt.Errorf("%s: managed credentials require the matching cloud endpoint", name)
}
