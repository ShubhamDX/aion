package config

import "testing"

func TestCloudCredentialConfiguration(t *testing.T) {
	for _, tt := range []struct {
		name, mode, endpoint, region, key string
		valid                             bool
	}{
		{"vertex", "google_adc", "", "us-east5", "", true},
		{"vertex", "google_adc", "", "user@other.invalid/#", "", false},
		{"vertex", "google_adc", "https://us-east5-aiplatform.googleapis.com", "", "", true},
		{"gemini", "google_adc", "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/openapi", "", "", true},
		{"gemini", "google_adc", "https://generativelanguage.googleapis.com/v1beta/openai", "", "", false},
		{"openai", "azure_entra", "https://example.openai.azure.com/openai/v1", "", "", true},
		{"openai", "azure_entra", "https://example.services.ai.azure.com/openai/v1", "", "", true},
		{"openai", "azure_entra", "https://example.openai.azure.com.attacker.invalid/openai/v1", "", "", false},
		{"openai", "azure_entra", "https://user:pass@example.openai.azure.com/openai/v1", "", "", false},
		{"openai", "azure_entra", "http://example.openai.azure.com/openai/v1", "", "", false},
		{"openai", "azure_entra", "https://example.openai.azure.com/openai/v1?api-version=old", "", "", false},
		{"openai", "azure_entra", "https://example.openai.azure.com/openai/v1", "", "ambiguous-key", false},
		{"vertex", "azure_entra", "", "", "", false},
		{"openai", "google_adc", "", "", "", false},
		{"vertex", "typo", "", "", "", false},
		{"openai", "", "http://127.0.0.1:8080/v1", "", "static-key", true},
	} {
		t.Run(tt.name+"/"+tt.mode+"/"+tt.endpoint+"/"+tt.region, func(t *testing.T) {
			err := ValidateCloudCredentials(tt.name, &ProviderConfig{
				CredentialMode: tt.mode, BaseURL: tt.endpoint, Region: tt.region, APIKey: tt.key,
				ProjectID: "fixture-project",
			})
			if (err == nil) != tt.valid {
				t.Fatalf("valid=%v error=%v", tt.valid, err)
			}
		})
	}
}

func TestManagedVertexRequiresInferenceProject(t *testing.T) {
	for _, project := range []string{"", "project/other"} {
		if err := ValidateCloudCredentials("vertex", &ProviderConfig{
			CredentialMode: "google_adc", ProjectID: project, Region: "us-east5",
		}); err == nil {
			t.Fatal("invalid inference project accepted")
		}
	}
}
