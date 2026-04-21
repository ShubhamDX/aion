package provider

import (
	"fmt"
	"os/exec"
	"strconv"
	"syscall"

	"github.com/ShubhamDX/aion/internal/config"
)

// LlamaProcess manages a llama-server subprocess for local model inference.
type LlamaProcess struct {
	cmd *exec.Cmd
	cfg *config.ManagedLlamaConfig
}

// NewLlamaProcess creates a new managed llama-server process (not yet started).
func NewLlamaProcess(cfg *config.ManagedLlamaConfig) *LlamaProcess {
	return &LlamaProcess{cfg: cfg}
}

// Start launches the llama-server subprocess.
func (lp *LlamaProcess) Start() error {
	args := lp.buildArgs()

	bin := lp.cfg.BinaryPath
	if bin == "" {
		bin = "llama-server"
	}

	lp.cmd = exec.Command(bin, args...)

	// Set process group so we can kill the entire group on shutdown.
	lp.cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := lp.cmd.Start(); err != nil {
		return fmt.Errorf("llama_process: start: %w", err)
	}
	return nil
}

// Stop sends SIGTERM to the llama-server process group and waits for exit.
func (lp *LlamaProcess) Stop() error {
	if lp.cmd == nil || lp.cmd.Process == nil {
		return nil
	}
	// Kill the entire process group.
	pgid, err := syscall.Getpgid(lp.cmd.Process.Pid)
	if err == nil {
		_ = syscall.Kill(-pgid, syscall.SIGTERM)
	}
	return lp.cmd.Wait()
}

func (lp *LlamaProcess) buildArgs() []string {
	var args []string

	args = append(args, "--model", lp.cfg.ModelPath)
	args = append(args, "--port", strconv.Itoa(lp.cfg.Port))

	if lp.cfg.ContextSize > 0 {
		args = append(args, "--ctx-size", strconv.Itoa(lp.cfg.ContextSize))
	}
	if lp.cfg.BatchSize > 0 {
		args = append(args, "--batch-size", strconv.Itoa(lp.cfg.BatchSize))
	}
	if lp.cfg.Threads > 0 {
		args = append(args, "--threads", strconv.Itoa(lp.cfg.Threads))
	}
	if lp.cfg.GPULayers > 0 {
		args = append(args, "--n-gpu-layers", strconv.Itoa(lp.cfg.GPULayers))
	}

	args = append(args, lp.cfg.ExtraArgs...)

	return args
}
