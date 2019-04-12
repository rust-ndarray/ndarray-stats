use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, Criterion,
    ParameterizedBenchmark, PlotConfiguration,
};
use ndarray::prelude::*;
use ndarray_stats::Sort1dExt;
use rand::prelude::*;

fn get_from_sorted_mut(c: &mut Criterion) {
    let lens = vec![10, 100, 1000, 10000];
    let benchmark = ParameterizedBenchmark::new(
        "get_from_sorted_mut",
        |bencher, &len| {
            let mut rng = StdRng::seed_from_u64(42);
            let mut data: Vec<_> = (0..len).collect();
            data.shuffle(&mut rng);
            let indices: Vec<_> = (0..len).step_by(len / 10).collect();
            bencher.iter_batched(
                || Array1::from(data.clone()),
                |mut arr| {
                    for &i in &indices {
                        black_box(arr.get_from_sorted_mut(i));
                    }
                },
                BatchSize::SmallInput,
            )
        },
        lens,
    )
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("get_from_sorted_mut", benchmark);
}

fn get_many_from_sorted_mut(c: &mut Criterion) {
    let lens = vec![10, 100, 1000, 10000];
    let benchmark = ParameterizedBenchmark::new(
        "get_many_from_sorted_mut",
        |bencher, &len| {
            let mut rng = StdRng::seed_from_u64(42);
            let mut data: Vec<_> = (0..len).collect();
            data.shuffle(&mut rng);
            let indices: Array1<_> = (0..len).step_by(len / 10).collect();
            bencher.iter_batched(
                || Array1::from(data.clone()),
                |mut arr| {
                    black_box(arr.get_many_from_sorted_mut(&indices));
                },
                BatchSize::SmallInput,
            )
        },
        lens,
    )
    .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    c.bench("get_many_from_sorted_mut", benchmark);
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = get_from_sorted_mut, get_many_from_sorted_mut
}
criterion_main!(benches);
