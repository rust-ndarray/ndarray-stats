use approx::{abs_diff_eq, assert_abs_diff_eq};
use ndarray::{array, Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::errors::EmptyInput;
use ndarray_stats::CorrelationExt;
use quickcheck_macros::quickcheck;

#[test]
fn perfect_monotonic_relationships_have_unit_coefficients() {
    let data = array![
        [1., 2., 3., 4., 5.],
        [10., 20., 30., 40., 50.],
        [50., 40., 30., 20., 10.]
    ];

    let spearman = data.spearman_correlation().unwrap();
    let kendall = data.kendall_tau().unwrap();
    assert_abs_diff_eq!(
        spearman,
        array![[1., 1., -1.], [1., 1., -1.], [-1., -1., 1.]],
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        kendall,
        array![[1., 1., -1.], [1., 1., -1.], [-1., -1., 1.]],
        epsilon = 1e-12
    );
}

#[test]
fn spearman_matches_known_rank_order_fixture() {
    let data = array![[1., 2., 3., 4., 5.], [5., 6., 7., 8., 7.]];
    let result = data.spearman_correlation().unwrap();
    assert_abs_diff_eq!(result[[0, 1]], 0.8207826816681233, epsilon = 1e-12);
}

#[test]
fn spearman_uses_average_ranks_for_ties() {
    let data = array![[1., 1., 2., 3., 3.], [1., 2., 2., 3., 4.]];
    let expected_ranks = array![[1.5, 1.5, 3., 4.5, 4.5], [1., 2.5, 2.5, 4., 5.]];
    let expected = expected_ranks.pearson_correlation().unwrap();

    assert_abs_diff_eq!(
        data.spearman_correlation().unwrap(),
        expected,
        epsilon = 1e-12
    );
}

#[test]
fn kendall_matches_known_tau_b_fixture() {
    let data = array![[12., 2., 1., 12., 2.], [1., 4., 7., 1., 0.]];
    let result = data.kendall_tau().unwrap();
    assert_abs_diff_eq!(result[[0, 1]], -0.47140452079103173, epsilon = 1e-12);
}

#[test]
fn kendall_handles_x_only_y_only_and_both_variable_ties() {
    let x_only = array![[1., 1., 2.], [1., 2., 3.]];
    let y_only = array![[1., 2., 3.], [1., 1., 2.]];
    let expected = 2. / 6_f64.sqrt();
    assert_abs_diff_eq!(
        x_only.kendall_tau().unwrap()[[0, 1]],
        expected,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        y_only.kendall_tau().unwrap()[[0, 1]],
        expected,
        epsilon = 1e-12
    );

    let both = array![[1., 1., 2., 2.], [1., 1., 2., 2.]];
    let reversed = array![[1., 1., 2., 2.], [2., 2., 1., 1.]];
    assert_abs_diff_eq!(both.kendall_tau().unwrap()[[0, 1]], 1., epsilon = 1e-12);
    assert_abs_diff_eq!(
        reversed.kendall_tau().unwrap()[[0, 1]],
        -1.,
        epsilon = 1e-12
    );
}

#[test]
fn rank_correlations_return_nan_for_constants_and_one_observation() {
    let data = array![[1_f64, 1., 1.], [1., 2., 3.]];
    let spearman = data.spearman_correlation().unwrap();
    let kendall = data.kendall_tau().unwrap();
    for result in [&spearman, &kendall] {
        assert!(result[[0, 0]].is_nan());
        assert!(result[[0, 1]].is_nan());
        assert!(result[[1, 0]].is_nan());
        assert_abs_diff_eq!(result[[1, 1]], 1., epsilon = 1e-12);
    }

    let one_observation = array![[1_f64], [2.]];
    assert!(one_observation.spearman_correlation().unwrap()[[0, 0]].is_nan());
    assert!(one_observation.kendall_tau().unwrap()[[0, 0]].is_nan());
}

#[test]
fn rank_correlations_return_empty_input_errors() {
    for data in [
        Array2::<f64>::zeros((0, 2)),
        Array2::<f64>::zeros((2, 0)),
        Array2::<f64>::zeros((0, 0)),
    ] {
        assert_eq!(data.spearman_correlation(), Err(EmptyInput));
        assert_eq!(data.kendall_tau(), Err(EmptyInput));
    }
}

#[test]
fn rank_correlations_propagate_nan_by_row() {
    let data = array![[1., f64::NAN, 3.], [1., 2., 3.], [3., 2., 1.]];
    let spearman = data.spearman_correlation().unwrap();
    let kendall = data.kendall_tau().unwrap();
    for result in [&spearman, &kendall] {
        assert!(result.row(0).iter().all(|value| value.is_nan()));
        assert!(result.column(0).iter().all(|value| value.is_nan()));
    }
    assert_abs_diff_eq!(spearman[[1, 2]], -1., epsilon = 1e-12);
    assert_abs_diff_eq!(kendall[[1, 2]], -1., epsilon = 1e-12);
}

#[test]
fn rank_correlations_do_not_modify_array_views() {
    let data = array![[4., 1., 3., 2.], [1., 2., 4., 3.]];
    let original = data.clone();
    let _ = data.view().spearman_correlation().unwrap();
    let _ = data.view().kendall_tau().unwrap();
    assert_eq!(data, original);
}

#[test]
fn rank_correlations_support_f32() {
    let data = array![[1_f32, 2., 3.], [3_f32, 2., 1.]];
    assert_abs_diff_eq!(
        data.spearman_correlation().unwrap()[[0, 1]],
        -1.,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(data.kendall_tau().unwrap()[[0, 1]], -1., epsilon = 1e-6);
}

#[quickcheck]
fn rank_correlation_matrices_are_symmetric_and_bounded(bound: f64) -> bool {
    if !bound.is_finite() {
        return true;
    }
    let bound = bound.abs() + 1.;
    let data = Array::random((3, 7), Uniform::new(-bound, bound).unwrap());
    let spearman = data.spearman_correlation().unwrap();
    let kendall = data.kendall_tau().unwrap();

    let symmetric = abs_diff_eq!(spearman.view(), spearman.t(), epsilon = 1e-8)
        && abs_diff_eq!(kendall.view(), kendall.t(), epsilon = 1e-8);
    let bounded = spearman
        .iter()
        .chain(kendall.iter())
        .all(|value| value.is_nan() || (*value >= -1. - 1e-12 && *value <= 1. + 1e-12));
    symmetric && bounded
}
