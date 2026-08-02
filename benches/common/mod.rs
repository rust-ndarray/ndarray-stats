use std::env;

/// Return the benchmark input sizes, optionally applying the CI size cap.
pub(crate) fn benchmark_lengths() -> Vec<usize> {
    const DEFAULT_LENGTHS: [usize; 4] = [10, 100, 1_000, 10_000];

    if let Ok(max_length) = env::var("BENCHMARK_MAX_INPUT_SIZE") {
        if let Ok(max_length) = max_length.parse::<usize>() {
            let lengths: Vec<_> = DEFAULT_LENGTHS
                .iter()
                .copied()
                .filter(|length| *length <= max_length)
                .collect();

            if !lengths.is_empty() {
                return lengths;
            }
        }
    }

    DEFAULT_LENGTHS.to_vec()
}
