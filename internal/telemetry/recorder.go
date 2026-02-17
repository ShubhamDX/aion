package telemetry

import (
	"context"
	"log"
	"time"
)

// Recorder is an asynchronous, buffered event recorder that batches
// RequestEvents and flushes them to the Store periodically or when the
// batch is full. It is designed to never block the request path.
type Recorder struct {
	store         *Store
	events        chan RequestEvent
	done          chan struct{}
	batchSize     int
	flushInterval time.Duration
}

// NewRecorder creates a Recorder with the given Store, batch size, and flush
// interval. The internal channel is sized to 10x batchSize to absorb bursts.
func NewRecorder(store *Store, batchSize int, flushInterval time.Duration) *Recorder {
	return &Recorder{
		store:         store,
		events:        make(chan RequestEvent, batchSize*10),
		done:          make(chan struct{}),
		batchSize:     batchSize,
		flushInterval: flushInterval,
	}
}

// Record enqueues an event for asynchronous persistence. If the internal
// buffer is full the event is silently dropped so that the caller (request
// handler) is never blocked.
func (r *Recorder) Record(event RequestEvent) {
	select {
	case r.events <- event:
	default:
		// Channel full — drop the event rather than blocking.
	}
}

// Start begins the background flush loop. It collects events from the channel
// and writes them in batches. The loop exits when ctx is cancelled or Stop is
// called.
func (r *Recorder) Start(ctx context.Context) {
	go r.loop(ctx)
}

func (r *Recorder) loop(ctx context.Context) {
	defer close(r.done)

	ticker := time.NewTicker(r.flushInterval)
	defer ticker.Stop()

	batch := make([]RequestEvent, 0, r.batchSize)

	flush := func() {
		if len(batch) == 0 {
			return
		}
		if err := r.store.InsertEvents(ctx, batch); err != nil {
			log.Printf("telemetry: flush %d events: %v", len(batch), err)
		}
		batch = batch[:0]
	}

	for {
		select {
		case ev, ok := <-r.events:
			if !ok {
				// Channel closed — flush remaining.
				flush()
				return
			}
			batch = append(batch, ev)
			if len(batch) >= r.batchSize {
				flush()
			}

		case <-ticker.C:
			flush()

		case <-ctx.Done():
			// Drain any remaining events in the channel.
			for {
				select {
				case ev, ok := <-r.events:
					if !ok {
						flush()
						return
					}
					batch = append(batch, ev)
				default:
					flush()
					return
				}
			}
		}
	}
}

// Stop closes the events channel and waits for the background goroutine to
// drain any remaining events and exit.
func (r *Recorder) Stop() {
	close(r.events)
	<-r.done
}
