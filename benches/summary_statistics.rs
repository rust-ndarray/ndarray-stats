use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BatchSize, Criterion, PlotConfiguration,
};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::{DescriptiveStatistics, QuantileExt, SummaryStatisticsExt};

mod common;

fn score_f64(summary: &DescriptiveStatistics<f64>) -> f64 {
    summary.count() as f64
        + summary.mean()
        + summary.min()
        + summary.max()
        + summary.population_variance()
        + summary.sample_variance()
        + summary.population_std()
        + summary.sample_std()
}

fn score_f32(summary: &DescriptiveStatistics<f32>) -> f32 {
    summary.count() as f32
        + summary.mean()
        + summary.min()
        + summary.max()
        + summary.population_variance()
        + summary.sample_variance()
        + summary.population_std()
        + summary.sample_std()
}

fn score_repeated_axis(data: &Array2<f64>, axis: Axis, weights: &Array1<f64>) -> f64 {
    data.lanes(axis)
        .into_iter()
        .map(|lane| {
            lane.mean().unwrap()
                + *lane.min().unwrap()
                + *lane.max().unwrap()
                + lane.weighted_var(weights, 0.0).unwrap()
                + lane.weighted_var(weights, 1.0).unwrap()
                + lane.weighted_std(weights, 0.0).unwrap()
                + lane.weighted_std(weights, 1.0).unwrap()
        })
        .sum()
}

fn weighted_std(c: &mut Criterion) {
    let lens = common::benchmark_lengths();
    let mut group = c.benchmark_group("weighted_std");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for len in &lens {
        group.bench_with_input(format!("{}", len), len, |b, &len| {
            let data = Array::random(len, Uniform::new(0.0, 1.0).unwrap());
            let mut weights = Array::random(len, Uniform::new(0.0, 1.0).unwrap());
            weights /= weights.sum();
            b.iter_batched(
                || data.clone(),
                |arr| {
                    black_box(arr.weighted_std(&weights, 0.0).unwrap());
                },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn descriptive_statistics(c: &mut Criterion) {
    let lens = common::benchmark_lengths();
    let mut group = c.benchmark_group("descriptive_statistics");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &len in &lens {
        let data = Array::random(len, Uniform::new(0.0, 1.0).unwrap());
        let weights = Array1::ones(len);

        group.bench_with_input(format!("fused/{len}"), &data, |b, data| {
            b.iter(|| {
                let summary = black_box(data.descriptive_statistics().unwrap());
                black_box(score_f64(&summary));
            })
        });

        group.bench_with_input(format!("repeated/{len}"), &data, |b, data| {
            b.iter(|| {
                let result = (
                    data.mean().unwrap(),
                    *data.min().unwrap(),
                    *data.max().unwrap(),
                    data.weighted_var(&weights, 0.0).unwrap(),
                    data.weighted_var(&weights, 1.0).unwrap(),
                    data.weighted_std(&weights, 0.0).unwrap(),
                    data.weighted_std(&weights, 1.0).unwrap(),
                );
                black_box(result);
            })
        });
    }

    group.finish();
}

fn descriptive_statistics_axis(c: &mut Criterion) {
    let lens = common::benchmark_lengths();
    let mut group = c.benchmark_group("descriptive_statistics_axis");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &len in &lens {
        let data = Array::random((len, 8), Uniform::new(0.0, 1.0).unwrap());
        let axis_zero_weights = Array1::ones(len);
        let axis_one_weights = Array1::ones(8);

        group.bench_with_input(format!("fused_axis0/{len}"), &data, |b, data| {
            b.iter(|| {
                let summaries = black_box(data.descriptive_statistics_axis(Axis(0)).unwrap());
                let score = summaries.iter().map(score_f64).sum::<f64>();
                black_box(score);
            })
        });

        group.bench_with_input(format!("repeated_axis0/{len}"), &data, |b, data| {
            b.iter(|| {
                black_box(score_repeated_axis(data, Axis(0), &axis_zero_weights));
            })
        });

        group.bench_with_input(format!("fused_axis1/{len}"), &data, |b, data| {
            b.iter(|| {
                let summaries = black_box(data.descriptive_statistics_axis(Axis(1)).unwrap());
                let score = summaries.iter().map(score_f64).sum::<f64>();
                black_box(score);
            })
        });

        group.bench_with_input(format!("repeated_axis1/{len}"), &data, |b, data| {
            b.iter(|| {
                black_box(score_repeated_axis(data, Axis(1), &axis_one_weights));
            })
        });
    }

    group.finish();
}

fn descriptive_statistics_f32(c: &mut Criterion) {
    let lens = common::benchmark_lengths();
    let mut group = c.benchmark_group("descriptive_statistics_f32");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &len in &lens {
        group.bench_function(format!("fused/{len}"), |b| {
            let data: Array1<f32> = Array::random(len, Uniform::new(0.0, 1.0).unwrap());
            b.iter(|| {
                let summary = black_box(data.descriptive_statistics().unwrap());
                black_box(score_f32(&summary));
            })
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = weighted_std, descriptive_statistics, descriptive_statistics_axis,
        descriptive_statistics_f32
}
criterion_main!(benches);
