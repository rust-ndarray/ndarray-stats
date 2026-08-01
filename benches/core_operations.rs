use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, Criterion, PlotConfiguration,
};
use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::histogram::{strategies::Auto, GridBuilder, HistogramExt};
use ndarray_stats::{
    interpolate::Linear, CorrelationExt, DeviationExt, EntropyExt, Quantile1dExt,
    SummaryStatisticsExt,
};
use noisy_float::types::n64;

fn mean(c: &mut Criterion) {
    let lens = vec![10, 100, 1000, 10000];
    let mut group = c.benchmark_group("mean");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for len in &lens {
        let data = Array::random(*len, Uniform::new(-1.0, 1.0).unwrap());
        let data_view = data.view();
        group.bench_with_input(format!("{}", len), len, |b, _| {
            b.iter(|| black_box(SummaryStatisticsExt::mean(&*data_view).unwrap()))
        });
    }
    group.finish();
}

fn quantiles_mut(c: &mut Criterion) {
    let lens = vec![10, 100, 1000, 10000];
    let quantile_indexes = Array1::from_vec(vec![n64(0.25), n64(0.5), n64(0.75)]);
    let mut group = c.benchmark_group("quantiles_mut");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for len in &lens {
        let data = Array1::from_iter((0..*len).rev().map(|value| value as i64));
        group.bench_with_input(format!("{}", len), len, |b, _| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    black_box(
                        data.quantiles_mut(&quantile_indexes.view(), &Linear)
                            .unwrap(),
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn pearson_correlation(c: &mut Criterion) {
    let lens = vec![10, 100, 1000, 10000];
    let mut group = c.benchmark_group("pearson_correlation");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for len in &lens {
        let data = Array::random((3, *len), Uniform::new(-1.0, 1.0).unwrap());
        group.bench_with_input(format!("{}", len), len, |b, _| {
            b.iter(|| black_box(data.pearson_correlation().unwrap()))
        });
    }
    group.finish();
}

fn entropy(c: &mut Criterion) {
    let lens = vec![10, 100, 1000, 10000];
    let mut group = c.benchmark_group("entropy");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for len in &lens {
        let mut data = Array::random(*len, Uniform::new(0.0, 1.0).unwrap());
        data /= data.sum();
        group.bench_with_input(format!("{}", len), len, |b, _| {
            b.iter(|| black_box(data.entropy().unwrap()))
        });
    }
    group.finish();
}

fn histogram(c: &mut Criterion) {
    let lens = vec![10, 100, 1000, 10000];
    let mut group = c.benchmark_group("histogram");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for len in &lens {
        let data = Array2::from_shape_fn((*len, 2), |(row, column)| {
            ((row * (column + 3) + column * 11) % 100) as i32
        });
        let grid = GridBuilder::<Auto<i32>>::from_array(&data).unwrap().build();
        group.bench_with_input(format!("{}", len), len, |b, _| {
            b.iter_batched(
                || (data.clone(), grid.clone()),
                |(data, grid)| black_box(data.histogram(grid)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn l1_dist(c: &mut Criterion) {
    let lens = vec![10, 100, 1000, 10000];
    let mut group = c.benchmark_group("l1_dist");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for len in &lens {
        let data = Array::random(*len, Uniform::new(0.0, 1.0).unwrap());
        let data2 = Array::random(*len, Uniform::new(0.0, 1.0).unwrap());
        group.bench_with_input(format!("{}", len), len, |b, _| {
            b.iter(|| black_box(data.l1_dist(&data2).unwrap()))
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = mean, quantiles_mut, pearson_correlation, entropy, histogram, l1_dist
}
criterion_main!(benches);
