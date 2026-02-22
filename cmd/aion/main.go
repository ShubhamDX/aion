package main

import (
	"flag"
	"log/slog"
	"os"

	"github.com/ShubhamDX/aion/pkg/app"
)

var version = "dev"

func main() {
	configPath := flag.String("config", "configs/aion.yaml", "path to AION config file")
	flag.Parse()

	slog.Info("starting AION", "version", version)
	app.Version = version

	application, err := app.Build(app.Options{
		ConfigPath: *configPath,
	})
	if err != nil {
		slog.Error("failed to build application", "error", err)
		os.Exit(1)
	}

	if err := application.Run(); err != nil {
		slog.Error("application error", "error", err)
		os.Exit(1)
	}
}
