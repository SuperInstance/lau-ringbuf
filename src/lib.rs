//! # lau-ringbuf
//!
//! Low-level lock-free-style circular buffer for real-time vibe data streaming.
//! Single-threaded with the API shape of a real-time audio buffer.


// ---------------------------------------------------------------------------
// RingBuf
// ---------------------------------------------------------------------------

/// Fixed-capacity circular buffer backed by an array of `T`.
///
/// `N` must be a power of two for correct modular arithmetic with `&`.
#[derive(Debug, Clone)]
pub struct RingBuf<T, const N: usize> {
    data: [T; N],
    write_head: usize,
    read_head: usize,
    count: usize,
}

impl<T: Clone + Default, const N: usize> RingBuf<T, N> {
    /// Create a new, empty ring buffer.
    pub fn new() -> Self {
        assert!(N > 0, "RingBuf size must be > 0");
        Self {
            data: std::array::from_fn(|_| T::default()),
            write_head: 0,
            read_head: 0,
            count: 0,
        }
    }

    /// Write an item. Returns `false` if the buffer is full.
    #[inline]
    pub fn write(&mut self, item: T) -> bool {
        if self.count == N {
            return false;
        }
        self.data[self.write_head] = item;
        self.write_head = (self.write_head + 1) % N;
        self.count += 1;
        true
    }

    /// Write an item, overwriting the oldest if the buffer is full.
    #[inline]
    pub fn write_overwrite(&mut self, item: T) {
        if self.count == N {
            // Advance read head to drop oldest
            self.read_head = (self.read_head + 1) % N;
        } else {
            self.count += 1;
        }
        self.data[self.write_head] = item;
        self.write_head = (self.write_head + 1) % N;
    }

    /// Read (consume) the oldest item.
    #[inline]
    pub fn read(&mut self) -> Option<T> {
        if self.count == 0 {
            return None;
        }
        let item = std::mem::take(&mut self.data[self.read_head]);
        self.read_head = (self.read_head + 1) % N;
        self.count -= 1;
        Some(item)
    }

    /// Peek at the oldest item without consuming it.
    #[inline]
    pub fn peek(&self) -> Option<&T> {
        if self.count == 0 {
            None
        } else {
            Some(&self.data[self.read_head])
        }
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.count == N
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Remaining write space.
    #[inline]
    pub fn available(&self) -> usize {
        N - self.count
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        for i in 0..N {
            self.data[i] = T::default();
        }
        self.write_head = 0;
        self.read_head = 0;
        self.count = 0;
    }
}

impl<T: Clone + Default, const N: usize> Default for RingBuf<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VibeRingBuf
// ---------------------------------------------------------------------------

/// Ring buffer specialized for `f64` vibe samples.
#[derive(Debug, Clone)]
pub struct VibeRingBuf<const N: usize> {
    inner: RingBuf<f64, N>,
}

impl<const N: usize> VibeRingBuf<N> {
    pub fn new() -> Self {
        Self {
            inner: RingBuf::new(),
        }
    }

    /// Push a vibe sample, overwriting oldest if full.
    #[inline]
    pub fn write_sample(&mut self, vibe: f64) {
        self.inner.write_overwrite(vibe);
    }

    /// Read the oldest sample.
    #[inline]
    pub fn read_sample(&mut self) -> Option<f64> {
        self.inner.read()
    }

    /// Collect the last `n` samples (newest last).
    pub fn recent(&self, n: usize) -> Vec<f64> {
        let n = n.min(self.inner.count);
        if n == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(n);
        // We need to iterate from (count - n) relative to read_head
        let start_offset = self.inner.count.saturating_sub(n);
        for i in 0..n {
            let idx = (self.inner.read_head + start_offset + i) % N;
            out.push(self.inner.data[idx]);
        }
        out
    }

    /// Mean of all buffered samples.
    pub fn average(&self) -> f64 {
        if self.inner.count == 0 {
            return 0.0;
        }
        let sum: f64 = self.iter_samples().sum();
        sum / self.inner.count as f64
    }

    /// Root mean square of all buffered samples.
    pub fn rms(&self) -> f64 {
        if self.inner.count == 0 {
            return 0.0;
        }
        let sum_sq: f64 = self.iter_samples().map(|s| s * s).sum();
        (sum_sq / self.inner.count as f64).sqrt()
    }

    /// Peak (max absolute value) of all buffered samples.
    pub fn peak(&self) -> f64 {
        if self.inner.count == 0 {
            return 0.0;
        }
        self.iter_samples().map(|s| s.abs()).fold(0.0_f64, f64::max)
    }

    /// Fraction of sign changes — a frequency proxy.
    pub fn zero_crossing_rate(&self) -> f64 {
        if self.inner.count < 2 {
            return 0.0;
        }
        let samples: Vec<f64> = self.iter_samples().collect();
        let crossings = samples.windows(2).filter(|w| w[0].signum() != w[1].signum()).count();
        crossings as f64 / (samples.len() - 1) as f64
    }

    /// First differences (derivative) of buffered samples.
    pub fn derivative(&self) -> Vec<f64> {
        if self.inner.count < 2 {
            return Vec::new();
        }
        let samples: Vec<f64> = self.iter_samples().collect();
        samples.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Number of buffered samples.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate samples in order (oldest first).
    fn iter_samples(&self) -> impl Iterator<Item = f64> + '_ {
        let count = self.inner.count;
        let read_head = self.inner.read_head;
        (0..count).map(move |i| self.inner.data[(read_head + i) % N])
    }

}

impl<const N: usize> Default for VibeRingBuf<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WindowedStats
// ---------------------------------------------------------------------------

/// Statistical summary of a sample window.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowedStats {
    pub mean: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

impl WindowedStats {
    /// Compute stats from a slice of samples.
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                mean: 0.0,
                variance: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
            };
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = if samples.len() > 1 {
            samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len().is_multiple_of(2) {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        Self {
            mean,
            variance,
            min: samples.iter().cloned().fold(f64::INFINITY, f64::min),
            max: samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            median,
        }
    }
}

// ---------------------------------------------------------------------------
// VibeStream
// ---------------------------------------------------------------------------

/// Ring buffer with sliding window analysis.
#[derive(Debug, Clone)]
pub struct VibeStream<const N: usize, const W: usize> {
    ring: VibeRingBuf<N>,
    window_size: usize,
}

impl<const N: usize, const W: usize> VibeStream<N, W> {
    pub fn new() -> Self {
        assert!(W <= N, "Window size W must be <= buffer size N");
        Self {
            ring: VibeRingBuf::new(),
            window_size: W,
        }
    }

    /// Push a sample into the stream.
    #[inline]
    pub fn push(&mut self, sample: f64) {
        self.ring.write_sample(sample);
    }

    /// Get the last `window_size` samples.
    fn window(&self) -> Vec<f64> {
        self.ring.recent(self.window_size)
    }

    /// Compute stats over the last W samples.
    pub fn window_stats(&self) -> WindowedStats {
        let w = self.window();
        WindowedStats::from_samples(&w)
    }

    /// Detect if current sample is > `threshold_sigma` standard deviations from mean.
    pub fn detect_anomaly(&self, threshold_sigma: f64) -> bool {
        let w = self.window();
        if w.len() < 2 {
            return false;
        }
        let stats = WindowedStats::from_samples(&w);
        let current = *w.last().unwrap();
        let stddev = stats.variance.sqrt();
        if stddev == 0.0 {
            return false;
        }
        ((current - stats.mean).abs() / stddev) > threshold_sigma
    }

    /// Linear regression slope of last W samples (trend).
    pub fn trend(&self) -> f64 {
        let w = self.window();
        if w.len() < 2 {
            return 0.0;
        }
        let n = w.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = w.iter().sum::<f64>() / n;

        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &y) in w.iter().enumerate() {
            let xi = i as f64 - x_mean;
            num += xi * (y - y_mean);
            den += xi * xi;
        }
        if den == 0.0 {
            0.0
        } else {
            num / den
        }
    }

    /// Read the oldest sample.
    pub fn read_sample(&mut self) -> Option<f64> {
        self.ring.read_sample()
    }
}

impl<const N: usize, const W: usize> Default for VibeStream<N, W> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- RingBuf basic tests --

    #[test]
    fn ringbuf_new_is_empty() {
        let buf: RingBuf<i32, 4> = RingBuf::new();
        assert!(buf.is_empty());
        assert!(!buf.is_full());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 4);
        assert_eq!(buf.available(), 4);
    }

    #[test]
    fn ringbuf_write_read() {
        let mut buf: RingBuf<i32, 4> = RingBuf::new();
        assert!(buf.write(1));
        assert!(buf.write(2));
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.read(), Some(1));
        assert_eq!(buf.read(), Some(2));
        assert_eq!(buf.read(), None);
    }

    #[test]
    fn ringbuf_full_rejects() {
        let mut buf: RingBuf<i32, 3> = RingBuf::new();
        assert!(buf.write(1));
        assert!(buf.write(2));
        assert!(buf.write(3));
        assert!(buf.is_full());
        assert!(!buf.write(4));
    }

    #[test]
    fn ringbuf_write_overwrite() {
        let mut buf: RingBuf<i32, 3> = RingBuf::new();
        buf.write_overwrite(1);
        buf.write_overwrite(2);
        buf.write_overwrite(3);
        buf.write_overwrite(4); // overwrites 1
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.read(), Some(2));
        assert_eq!(buf.read(), Some(3));
        assert_eq!(buf.read(), Some(4));
        assert_eq!(buf.read(), None);
    }

    #[test]
    fn ringbuf_peek() {
        let mut buf: RingBuf<i32, 4> = RingBuf::new();
        assert_eq!(buf.peek(), None);
        buf.write(42);
        assert_eq!(buf.peek(), Some(&42));
        assert_eq!(buf.len(), 1); // peek doesn't consume
    }

    #[test]
    fn ringbuf_clear() {
        let mut buf: RingBuf<i32, 4> = RingBuf::new();
        buf.write(1);
        buf.write(2);
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.available(), 4);
    }

    #[test]
    fn ringbuf_wrap_around() {
        let mut buf: RingBuf<i32, 3> = RingBuf::new();
        buf.write(1);
        buf.write(2);
        buf.read(); // remove 1
        buf.write(3);
        buf.write(4); // wraps around
        assert_eq!(buf.read(), Some(2));
        assert_eq!(buf.read(), Some(3));
        assert_eq!(buf.read(), Some(4));
    }

    #[test]
    fn ringbuf_default() {
        let buf: RingBuf<i32, 4> = RingBuf::default();
        assert!(buf.is_empty());
    }

    // -- VibeRingBuf tests --

    #[test]
    fn vibe_write_read() {
        let mut buf: VibeRingBuf<8> = VibeRingBuf::new();
        buf.write_sample(1.0);
        buf.write_sample(2.5);
        assert_eq!(buf.read_sample(), Some(1.0));
        assert_eq!(buf.read_sample(), Some(2.5));
        assert_eq!(buf.read_sample(), None);
    }

    #[test]
    fn vibe_recent() {
        let mut buf: VibeRingBuf<8> = VibeRingBuf::new();
        for i in 0..5 {
            buf.write_sample(i as f64);
        }
        assert_eq!(buf.recent(3), vec![2.0, 3.0, 4.0]);
        assert_eq!(buf.recent(10), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(buf.recent(0), Vec::<f64>::new());
    }

    #[test]
    fn vibe_average() {
        let mut buf: VibeRingBuf<8> = VibeRingBuf::new();
        assert_eq!(buf.average(), 0.0);
        buf.write_sample(2.0);
        buf.write_sample(4.0);
        buf.write_sample(6.0);
        assert!((buf.average() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn vibe_rms() {
        let mut buf: VibeRingBuf<8> = VibeRingBuf::new();
        buf.write_sample(3.0);
        buf.write_sample(4.0);
        // sqrt((9+16)/2) = sqrt(12.5) ≈ 3.535
        let expected = (12.5_f64).sqrt();
        assert!((buf.rms() - expected).abs() < 1e-9);
    }

    #[test]
    fn vibe_peak() {
        let mut buf: VibeRingBuf<8> = VibeRingBuf::new();
        buf.write_sample(1.0);
        buf.write_sample(-5.0);
        buf.write_sample(3.0);
        assert!((buf.peak() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn vibe_zero_crossing_rate() {
        let mut buf: VibeRingBuf<8> = VibeRingBuf::new();
        buf.write_sample(1.0);
        buf.write_sample(-1.0);
        buf.write_sample(1.0);
        buf.write_sample(-1.0);
        // 3 crossings in 3 intervals
        assert!((buf.zero_crossing_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn vibe_derivative() {
        let mut buf: VibeRingBuf<8> = VibeRingBuf::new();
        buf.write_sample(1.0);
        buf.write_sample(3.0);
        buf.write_sample(6.0);
        assert_eq!(buf.derivative(), vec![2.0, 3.0]);
    }

    #[test]
    fn vibe_overwrite_keeps_latest() {
        let mut buf: VibeRingBuf<3> = VibeRingBuf::new();
        buf.write_sample(1.0);
        buf.write_sample(2.0);
        buf.write_sample(3.0);
        buf.write_sample(4.0); // overwrites 1.0
        assert_eq!(buf.recent(3), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn vibe_empty_stats() {
        let buf: VibeRingBuf<8> = VibeRingBuf::new();
        assert_eq!(buf.average(), 0.0);
        assert_eq!(buf.rms(), 0.0);
        assert_eq!(buf.peak(), 0.0);
        assert_eq!(buf.zero_crossing_rate(), 0.0);
        assert!(buf.derivative().is_empty());
    }

    // -- WindowedStats tests --

    #[test]
    fn windowed_stats_basic() {
        let stats = WindowedStats::from_samples(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((stats.mean - 3.0).abs() < 1e-9);
        assert!((stats.min - 1.0).abs() < 1e-9);
        assert!((stats.max - 5.0).abs() < 1e-9);
        assert!((stats.median - 3.0).abs() < 1e-9);
    }

    #[test]
    fn windowed_stats_variance() {
        let stats = WindowedStats::from_samples(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        // sample variance = 4.571428...
        let expected = 32.0 / 7.0;
        assert!((stats.variance - expected).abs() < 1e-6);
    }

    #[test]
    fn windowed_stats_even_median() {
        let stats = WindowedStats::from_samples(&[1.0, 3.0, 5.0, 7.0]);
        assert!((stats.median - 4.0).abs() < 1e-9);
    }

    #[test]
    fn windowed_stats_empty() {
        let stats = WindowedStats::from_samples(&[]);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.variance, 0.0);
        assert_eq!(stats.median, 0.0);
    }

    #[test]
    fn windowed_stats_single() {
        let stats = WindowedStats::from_samples(&[42.0]);
        assert!((stats.mean - 42.0).abs() < 1e-9);
        assert_eq!(stats.variance, 0.0);
        assert!((stats.median - 42.0).abs() < 1e-9);
    }

    // -- VibeStream tests --

    #[test]
    fn vibe_stream_push_and_stats() {
        let mut stream: VibeStream<16, 4> = VibeStream::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            stream.push(v);
        }
        let stats = stream.window_stats();
        // window is [2,3,4,5]
        assert!((stats.mean - 3.5).abs() < 1e-9);
        assert!((stats.min - 2.0).abs() < 1e-9);
        assert!((stats.max - 5.0).abs() < 1e-9);
    }

    #[test]
    fn vibe_stream_anomaly_detect() {
        let mut stream: VibeStream<16, 10> = VibeStream::new();
        for _ in 0..9 {
            stream.push(1.0);
        }
        // All same values, no anomaly
        assert!(!stream.detect_anomaly(2.0));
        stream.push(1.0);
        assert!(!stream.detect_anomaly(2.0));
    }

    #[test]
    fn vibe_stream_anomaly_triggered() {
        let mut stream: VibeStream<16, 10> = VibeStream::new();
        for _ in 0..9 {
            stream.push(0.0);
        }
        stream.push(100.0);
        assert!(stream.detect_anomaly(2.0));
    }

    #[test]
    fn vibe_stream_trend_positive() {
        let mut stream: VibeStream<16, 5> = VibeStream::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            stream.push(v);
        }
        let t = stream.trend();
        assert!(t > 0.0, "trend should be positive, got {t}");
        assert!((t - 1.0).abs() < 1e-9, "slope should be ~1.0, got {t}");
    }

    #[test]
    fn vibe_stream_trend_flat() {
        let mut stream: VibeStream<16, 5> = VibeStream::new();
        for _ in 0..5 {
            stream.push(3.0);
        }
        let t = stream.trend();
        assert!(t.abs() < 1e-9, "flat trend should be ~0, got {t}");
    }

    #[test]
    fn vibe_stream_trend_negative() {
        let mut stream: VibeStream<16, 5> = VibeStream::new();
        for v in [5.0, 4.0, 3.0, 2.0, 1.0] {
            stream.push(v);
        }
        let t = stream.trend();
        assert!(t < 0.0, "trend should be negative, got {t}");
    }

    #[test]
    fn vibe_stream_trend_empty() {
        let stream: VibeStream<16, 5> = VibeStream::new();
        assert_eq!(stream.trend(), 0.0);
    }
}
