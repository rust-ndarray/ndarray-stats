use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, ParameterizedBenchmark,
    PlotConfiguration,
};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_stats::DeviationExt;
use ndarray_rand::rand_distr::Uniform;

fn sq_l2_dist(c: &mut Criterion) {
    let lens = vec![10, 100, 1000, 10000];
    let benchmark = ParameterizedBenchmark::new(
        "sq_l2_dist",
        |bencher, &len| {
            let data = Array::random(len, Uniform::new(0.0, 1.0));
            let data2 = Array::random(len, Uniform::new(0.0, 1.0));

            bencher.iter(|| black_box(data.sq_l2_dist(&data2).unwrap()))
        },
        lens,
    )
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("sq_l2_dist", benchmark);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = sq_l2_dist
}
criterion_main!(benches);
