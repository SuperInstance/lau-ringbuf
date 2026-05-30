# lau-ringbuf

Low-level lock-free-style circular buffer for real-time vibe data streaming. Single-threaded with the API shape of a real-time audio buffer.

## Features

- **`RingBuf<T, N>`** — Fixed-capacity circular buffer with const generic size
- **`VibeRingBuf<N>`** — Specialized for `f64` vibe samples with signal analysis
- **`WindowedStats`** — Statistical summary (mean, variance, min, max, median)
- **`VibeStream<N, W>`** — Ring buffer with sliding window analysis and anomaly detection

## Usage

```rust
use lau_ringbuf::{RingBuf, VibeRingBuf, VibeStream};

// Basic ring buffer
let mut buf: RingBuf<i32, 16> = RingBuf::new();
buf.write(42);
assert_eq!(buf.read(), Some(42));

// Vibe sample buffer
let mut vibes: VibeRingBuf<256> = VibeRingBuf::new();
vibes.write_sample(0.75);
println!("rms: {}", vibes.rms());

// Streaming analysis with sliding window
let mut stream: VibeStream<1024, 64> = VibeStream::new();
stream.push(1.0);
let stats = stream.window_stats();
let anomaly = stream.detect_anomaly(2.0);
let trend = stream.trend();
```
