use approx::assert_abs_diff_eq;
use ndarray::{array, Axis};
use ndarray_stats::{
    errors::{NonFiniteValue, SummaryStatisticsError},
    InfinityPolicy, MissingDataPolicy, NumericPolicy, SummaryStatisticsExt,
};

#[test]
fn omission_ignores_nan_and_preserves_observation_count() {
    let data = array![1.0, f64::NAN, 3.0];

    let summary = data
        .descriptive_statistics_with_policy(NumericPolicy::omit_missing())
        .unwrap();

    assert_eq!(summary.count(), 2);
    assert_abs_diff_eq!(summary.mean(), 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(summary.population_variance(), 1.0, epsilon = 1e-12);
    assert_eq!(summary.min(), 1.0);
    assert_eq!(summary.max(), 3.0);
}

#[test]
fn omission_is_applied_independently_to_each_axis_lane() {
    let data = array![[1.0, f64::NAN, 3.0], [f64::NAN, 5.0, 7.0]];

    let summaries = data
        .descriptive_statistics_axis_with_policy(Axis(1), NumericPolicy::omit_missing())
        .unwrap();

    assert_eq!(summaries.shape(), &[2]);
    assert_eq!(summaries[0].count(), 2);
    assert_abs_diff_eq!(summaries[0].mean(), 2.0, epsilon = 1e-12);
    assert_eq!(summaries[1].count(), 2);
    assert_abs_diff_eq!(summaries[1].mean(), 6.0, epsilon = 1e-12);
}

#[test]
fn omission_of_every_value_is_an_empty_input() {
    let data = array![f64::NAN, f64::NAN];

    assert_eq!(
        data.descriptive_statistics_with_policy(NumericPolicy::omit_missing()),
        Err(SummaryStatisticsError::EmptyInput)
    );
}

#[test]
fn omission_of_every_value_in_one_axis_lane_is_an_empty_input() {
    let data = array![[1.0, 2.0], [f64::NAN, f64::NAN]];

    assert_eq!(
        data.descriptive_statistics_axis_with_policy(Axis(1), NumericPolicy::omit_missing()),
        Err(SummaryStatisticsError::EmptyInput)
    );
}

#[test]
fn omission_does_not_silently_remove_infinity() {
    let data = array![1.0, f64::INFINITY, 3.0];

    let summary = data
        .descriptive_statistics_with_policy(NumericPolicy::omit_missing())
        .unwrap();

    assert_eq!(summary.count(), 3);
    // Infinity is retained. The accumulator then encounters `∞ - ∞`, so the
    // mean is indeterminate rather than silently behaving as if infinity were
    // missing.
    assert!(summary.mean().is_nan());
    assert!(summary.population_variance().is_nan());
}

#[test]
fn reject_policy_reports_nan_and_infinity() {
    let reject_nan = NumericPolicy::new(MissingDataPolicy::Reject, InfinityPolicy::Propagate);
    let nan_result = array![1.0, f64::NAN].descriptive_statistics_with_policy(reject_nan);
    assert_eq!(
        nan_result,
        Err(SummaryStatisticsError::NonFiniteValue {
            index: 1,
            value: NonFiniteValue::Nan,
        })
    );

    let infinity_result = array![1.0, f64::NEG_INFINITY]
        .descriptive_statistics_with_policy(NumericPolicy::reject_non_finite());
    assert_eq!(
        infinity_result,
        Err(SummaryStatisticsError::NonFiniteValue {
            index: 1,
            value: NonFiniteValue::NegativeInfinity,
        })
    );
}

#[test]
fn default_policy_preserves_existing_nan_behavior() {
    let data = array![1.0, f64::NAN];

    assert_eq!(
        data.descriptive_statistics_with_policy(NumericPolicy::default()),
        data.descriptive_statistics()
    );
}
