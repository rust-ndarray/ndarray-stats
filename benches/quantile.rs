extern crate criterion;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_stats;
extern crate noisy_float;
extern crate rand;

use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, Criterion, ParameterizedBenchmark,
    PlotConfiguration,
};
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_stats::{interpolate::Midpoint, Quantile1dExt, QuantileExt};
use noisy_float::types::n64;
use rand::distributions::Uniform;
use rand::random;

fn quantile_mut(c: &mut Criterion) {
    let sizes = vec![10, 100, 1_000, 10_000];
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    c.bench(
        "quantile_mut",
        ParameterizedBenchmark::new(
            "quantile_mut",
            |bencher, &size| {
                bencher.iter_with_setup(
                    || {
                        (
                            Array1::random(size, Uniform::new(-50., 50.)).mapv(n64),
                            random(),
                        )
                    },
                    |(mut arr, q)| black_box(arr.quantile_mut::<Midpoint>(q)),
                )
            },
            sizes,
        )
        .plot_config(plot_config),
    );
}

fn min_and_min_skipnan_without_nans(c: &mut Criterion) {
    let sizes = vec![10, 100, 1_000, 10_000];
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    c.bench(
        "min_and_min_skipnan_without_nans",
        ParameterizedBenchmark::new(
            "min",
            |bencher, &size| {
                bencher.iter_with_setup(
                    || Array1::random(size, Uniform::new(-50., 50.)),
                    |arr| black_box(*arr.min().unwrap()),
                )
            },
            sizes,
        )
        .with_function("min_skipnan", |bencher, &size| {
            bencher.iter_with_setup(
                || Array1::random(size, Uniform::new(-50., 50.)),
                |arr| black_box(*arr.min_skipnan()),
            )
        })
        .plot_config(plot_config),
    );
}

fn quantile_axis_mut(c: &mut Criterion) {
    let sizes = vec![10, 100, 1_000];
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    c.bench(
        "quantile_axis_mut",
        ParameterizedBenchmark::new(
            "rowwise",
            |bencher, &size| {
                bencher.iter_with_setup(
                    || {
                        (
                            Array2::random((size, 20), Uniform::new(-50., 50.)).mapv(n64),
                            random(),
                        )
                    },
                    |(mut arr, q)| black_box(arr.quantile_axis_mut::<Midpoint>(Axis(0), q)),
                )
            },
            sizes,
        )
        .with_function("columnwise", |bencher, &size| {
            bencher.iter_with_setup(
                || {
                    (
                        Array2::random((20, size), Uniform::new(-50., 50.)).mapv(n64),
                        random(),
                    )
                },
                |(mut arr, q)| black_box(arr.quantile_axis_mut::<Midpoint>(Axis(1), q)),
            )
        })
        .plot_config(plot_config),
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = quantile_mut, min_and_min_skipnan_without_nans, quantile_axis_mut
}
criterion_main!(benches);
