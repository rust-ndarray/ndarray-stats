use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, Criterion,
    ParameterizedBenchmark, PlotConfiguration,
};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_stats::SummaryStatisticsExt;
use ndarray_rand::rand_distr::Uniform;

fn weighted_std(c: &mut Criterion) {
    let lens = vec![10, 100, 1000, 10000];
    let benchmark = ParameterizedBenchmark::new(
        "weighted_std",
        |bencher, &len| {
            let data = Array::random(len, Uniform::new(0.0, 1.0));
            let mut weights = Array::random(len, Uniform::new(0.0, 1.0));
            weights /= weights.sum();
            bencher.iter_batched(
                || data.clone(),
                |arr| {
                    black_box(arr.weighted_std(&weights, 0.0).unwrap());
                },
                BatchSize::SmallInput,
            )
        },
        lens,
    )
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("weighted_std", benchmark);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = weighted_std
}
criterion_main!(benches);
