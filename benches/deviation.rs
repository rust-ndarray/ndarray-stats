use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, PlotConfiguration,
};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::DeviationExt;

fn sq_l2_dist(c: &mut Criterion) {
    let lens = vec![10, 100, 1000, 10000];
    let mut group = c.benchmark_group("sq_l2_dist");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for len in &lens {
        group.bench_with_input(format!("{}", len), len, |b, &len| {
            let data = Array::random(len, Uniform::new(0.0, 1.0));
            let data2 = Array::random(len, Uniform::new(0.0, 1.0));

            b.iter(|| black_box(data.sq_l2_dist(&data2).unwrap()))
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = sq_l2_dist
}
criterion_main!(benches);
