package provider

import (
	"fmt"
	"log/slog"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/ShubhamDX/aion/internal/config"
)

// stopGrace is how long we wait for SIGTERM to take effect before SIGKILL.
const stopGrace = 10 * time.Second

// LlamaProcess manages a llama-server subprocess for local model inference.
type LlamaProcess struct {
	cmd      *exec.Cmd
	cfg      *config.ManagedLlamaConfig
	stopOnce sync.Once
	stopErr  error
	exited   chan struct{}
}

// NewLlamaProcess creates a new managed llama-server process (not yet started).
func NewLlamaProcess(cfg *config.ManagedLlamaConfig) *LlamaProcess {
	return &LlamaProcess{cfg: cfg, exited: make(chan struct{})}
}

// Start launches the llama-server subprocess. It also starts a background
// reaper goroutine that closes lp.exited once the process is gone, so Stop
// can safely wait with a timeout and never block twice on the same cmd.
func (lp *LlamaProcess) Start() error {
	args := lp.buildArgs()

	bin := lp.cfg.BinaryPath
	if bin == "" {
		bin = "llama-server"
	}

	lp.cmd = exec.Command(bin, args...)
	lp.cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := lp.cmd.Start(); err != nil {
		return fmt.Errorf("llama_process: start: %w", err)
	}

	// Reap the process in the background so Wait() is only called once.
	go func() {
		_ = lp.cmd.Wait()
		close(lp.exited)
	}()
	return nil
}

// Stop sends SIGTERM to the llama-server process group, waits up to stopGrace
// for it to exit, then escalates to SIGKILL. Idempotent.
func (lp *LlamaProcess) Stop() error {
	lp.stopOnce.Do(func() {
		if lp.cmd == nil || lp.cmd.Process == nil {
			return
		}
		// Fast path: already dead.
		select {
		case <-lp.exited:
			return
		default:
		}

		if pgid, err := syscall.Getpgid(lp.cmd.Process.Pid); err == nil {
			_ = syscall.Kill(-pgid, syscall.SIGTERM)
		}

		select {
		case <-lp.exited:
			return
		case <-time.After(stopGrace):
			slog.Warn("llama-server did not exit on SIGTERM, sending SIGKILL")
			if pgid, err := syscall.Getpgid(lp.cmd.Process.Pid); err == nil {
				_ = syscall.Kill(-pgid, syscall.SIGKILL)
			} else {
				_ = lp.cmd.Process.Kill()
			}
			<-lp.exited
		}
	})
	return lp.stopErr
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
