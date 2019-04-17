use ndarray_stats::errors::{MultiInputError, ShapeMismatch};
use ndarray_stats::DeviationExt;

use approx::assert_abs_diff_eq;
use ndarray::{array, Array1};
use num_bigint::BigInt;
use num_traits::Float;

use std::f64;

#[test]
fn test_count_eq() -> Result<(), MultiInputError> {
    let a = array![0., 0.];
    let b = array![1., 0.];
    let c = array![0., 1.];
    let d = array![1., 1.];

    assert_eq!(a.count_eq(&a)?, 2);
    assert_eq!(a.count_eq(&b)?, 1);
    assert_eq!(a.count_eq(&c)?, 1);
    assert_eq!(a.count_eq(&d)?, 0);

    Ok(())
}

#[test]
fn test_count_neq() -> Result<(), MultiInputError> {
    let a = array![0., 0.];
    let b = array![1., 0.];
    let c = array![0., 1.];
    let d = array![1., 1.];

    assert_eq!(a.count_neq(&a)?, 0);
    assert_eq!(a.count_neq(&b)?, 1);
    assert_eq!(a.count_neq(&c)?, 1);
    assert_eq!(a.count_neq(&d)?, 2);

    Ok(())
}

#[test]
fn test_sq_l2_dist() -> Result<(), MultiInputError> {
    let a = array![0., 1., 4., 2.];
    let b = array![1., 1., 2., 4.];

    assert_eq!(a.sq_l2_dist(&b)?, 9.);

    Ok(())
}

#[test]
fn test_l2_dist() -> Result<(), MultiInputError> {
    let a = array![0., 1., 4., 2.];
    let b = array![1., 1., 2., 4.];

    assert_eq!(a.l2_dist(&b)?, 3.);

    Ok(())
}

#[test]
fn test_l1_dist() -> Result<(), MultiInputError> {
    let a = array![0., 1., 4., 2.];
    let b = array![1., 1., 2., 4.];

    assert_eq!(a.l1_dist(&b)?, 5.);

    Ok(())
}

#[test]
fn test_linf_dist() -> Result<(), MultiInputError> {
    let a = array![0., 0.];
    let b = array![1., 0.];
    let c = array![1., 2.];

    assert_eq!(a.linf_dist(&a)?, 0.);

    assert_eq!(a.linf_dist(&b)?, 1.);
    assert_eq!(b.linf_dist(&a)?, 1.);

    assert_eq!(a.linf_dist(&c)?, 2.);
    assert_eq!(c.linf_dist(&a)?, 2.);

    Ok(())
}

#[test]
fn test_mean_abs_err() -> Result<(), MultiInputError> {
    let a = array![1., 1.];
    let b = array![3., 5.];

    assert_eq!(a.mean_abs_err(&a)?, 0.);
    assert_eq!(a.mean_abs_err(&b)?, 3.);
    assert_eq!(b.mean_abs_err(&a)?, 3.);

    Ok(())
}

#[test]
fn test_mean_sq_err() -> Result<(), MultiInputError> {
    let a = array![1., 1.];
    let b = array![3., 5.];

    assert_eq!(a.mean_sq_err(&a)?, 0.);
    assert_eq!(a.mean_sq_err(&b)?, 10.);
    assert_eq!(b.mean_sq_err(&a)?, 10.);

    Ok(())
}

#[test]
fn test_root_mean_sq_err() -> Result<(), MultiInputError> {
    let a = array![1., 1.];
    let b = array![3., 5.];

    assert_eq!(a.root_mean_sq_err(&a)?, 0.);
    assert_abs_diff_eq!(a.root_mean_sq_err(&b)?, 10.0.sqrt());
    assert_abs_diff_eq!(b.root_mean_sq_err(&a)?, 10.0.sqrt());

    Ok(())
}

#[test]
fn test_peak_signal_to_noise_ratio() -> Result<(), MultiInputError> {
    let a = array![1., 1.];
    assert!(a.peak_signal_to_noise_ratio(&a, 1.)?.is_infinite());

    let a = array![1., 2., 3., 4., 5., 6., 7.];
    let b = array![1., 3., 3., 4., 6., 7., 8.];
    let maxv = 8.;
    let expected = 20. * Float::log10(maxv) - 10. * Float::log10(a.mean_sq_err(&b)?);
    let actual = a.peak_signal_to_noise_ratio(&b, maxv)?;

    assert_abs_diff_eq!(actual, expected);

    Ok(())
}

#[test]
fn test_deviations_with_n_by_m_ints() -> Result<(), MultiInputError> {
    let a = array![[0, 1], [4, 2]];
    let b = array![[1, 1], [2, 4]];

    assert_eq!(a.count_eq(&a)?, 4);
    assert_eq!(a.count_neq(&a)?, 0);

    assert_eq!(a.sq_l2_dist(&b)?, 9);
    assert_eq!(a.l2_dist(&b)?, 3.);
    assert_eq!(a.l1_dist(&b)?, 5);
    assert_eq!(a.linf_dist(&b)?, 2);

    assert_abs_diff_eq!(a.mean_abs_err(&b)?, 1.25);
    assert_abs_diff_eq!(a.mean_sq_err(&b)?, 2.25);
    assert_abs_diff_eq!(a.root_mean_sq_err(&b)?, 1.5);
    assert_abs_diff_eq!(a.peak_signal_to_noise_ratio(&b, 4)?, 8.519374645445623);

    Ok(())
}

#[test]
fn test_deviations_with_empty_receiver() {
    let a: Array1<f64> = array![];
    let b: Array1<f64> = array![1.];

    assert_eq!(a.count_eq(&b), Err(MultiInputError::EmptyInput));
    assert_eq!(a.count_neq(&b), Err(MultiInputError::EmptyInput));

    assert_eq!(a.sq_l2_dist(&b), Err(MultiInputError::EmptyInput));
    assert_eq!(a.l2_dist(&b), Err(MultiInputError::EmptyInput));
    assert_eq!(a.l1_dist(&b), Err(MultiInputError::EmptyInput));
    assert_eq!(a.linf_dist(&b), Err(MultiInputError::EmptyInput));

    assert_eq!(a.mean_abs_err(&b), Err(MultiInputError::EmptyInput));
    assert_eq!(a.mean_sq_err(&b), Err(MultiInputError::EmptyInput));
    assert_eq!(a.root_mean_sq_err(&b), Err(MultiInputError::EmptyInput));
    assert_eq!(
        a.peak_signal_to_noise_ratio(&b, 0.),
        Err(MultiInputError::EmptyInput)
    );
}

#[test]
fn test_deviations_do_not_panic_if_nans() -> Result<(), MultiInputError> {
    let a: Array1<f64> = array![1., f64::NAN, 3., f64::NAN];
    let b: Array1<f64> = array![1., f64::NAN, 3., 4.];

    assert_eq!(a.count_eq(&b)?, 2);
    assert_eq!(a.count_neq(&b)?, 2);

    assert!(a.sq_l2_dist(&b)?.is_nan());
    assert!(a.l2_dist(&b)?.is_nan());
    assert!(a.l1_dist(&b)?.is_nan());
    assert_eq!(a.linf_dist(&b)?, 0.);

    assert!(a.mean_abs_err(&b)?.is_nan());
    assert!(a.mean_sq_err(&b)?.is_nan());
    assert!(a.root_mean_sq_err(&b)?.is_nan());
    assert!(a.peak_signal_to_noise_ratio(&b, 0.)?.is_nan());

    Ok(())
}

#[test]
fn test_deviations_with_empty_argument() {
    let a: Array1<f64> = array![1.];
    let b: Array1<f64> = array![];

    let shape_mismatch_err = MultiInputError::ShapeMismatch(ShapeMismatch {
        first_shape: a.shape().to_vec(),
        second_shape: b.shape().to_vec(),
    });
    let expected_err_usize = Err(shape_mismatch_err.clone());
    let expected_err_f64 = Err(shape_mismatch_err);

    assert_eq!(a.count_eq(&b), expected_err_usize);
    assert_eq!(a.count_neq(&b), expected_err_usize);

    assert_eq!(a.sq_l2_dist(&b), expected_err_f64);
    assert_eq!(a.l2_dist(&b), expected_err_f64);
    assert_eq!(a.l1_dist(&b), expected_err_f64);
    assert_eq!(a.linf_dist(&b), expected_err_f64);

    assert_eq!(a.mean_abs_err(&b), expected_err_f64);
    assert_eq!(a.mean_sq_err(&b), expected_err_f64);
    assert_eq!(a.root_mean_sq_err(&b), expected_err_f64);
    assert_eq!(a.peak_signal_to_noise_ratio(&b, 0.), expected_err_f64);
}

#[test]
fn test_deviations_with_non_copyable() -> Result<(), MultiInputError> {
    let a: Array1<BigInt> = array![0.into(), 1.into(), 4.into(), 2.into()];
    let b: Array1<BigInt> = array![1.into(), 1.into(), 2.into(), 4.into()];

    assert_eq!(a.count_eq(&a)?, 4);
    assert_eq!(a.count_neq(&a)?, 0);

    assert_eq!(a.sq_l2_dist(&b)?, 9.into());
    assert_eq!(a.l2_dist(&b)?, 3.);
    assert_eq!(a.l1_dist(&b)?, 5.into());
    assert_eq!(a.linf_dist(&b)?, 2.into());

    assert_abs_diff_eq!(a.mean_abs_err(&b)?, 1.25);
    assert_abs_diff_eq!(a.mean_sq_err(&b)?, 2.25);
    assert_abs_diff_eq!(a.root_mean_sq_err(&b)?, 1.5);
    assert_abs_diff_eq!(
        a.peak_signal_to_noise_ratio(&b, 4.into())?,
        8.519374645445623
    );

    Ok(())
}
