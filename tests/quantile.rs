extern crate ndarray;
extern crate ndarray_stats;
extern crate noisy_float;
#[macro_use]
extern crate quickcheck;
extern crate quickcheck_macros;

use noisy_float::types::{n64, N64};
use ndarray::prelude::*;
use ndarray::array;
use ndarray_stats::{
    interpolate::{Interpolate, Higher, Linear, Lower, Midpoint, Nearest},
    Quantile1dExt, QuantileExt,
};
use quickcheck_macros::quickcheck;

#[test]
fn test_argmin() {
    let a = array![[1, 5, 3], [2, 0, 6]];
    assert_eq!(a.argmin(), Some((1, 1)));

    let a = array![[1., 5., 3.], [2., 0., 6.]];
    assert_eq!(a.argmin(), Some((1, 1)));

    let a = array![[1., 5., 3.], [2., ::std::f64::NAN, 6.]];
    assert_eq!(a.argmin(), None);

    let a: Array2<i32> = array![[], []];
    assert_eq!(a.argmin(), None);
}

quickcheck! {
    fn argmin_matches_min(data: Vec<f32>) -> bool {
        let a = Array1::from(data);
        a.argmin().map(|i| a[i]) == a.min().cloned()
    }
}

#[test]
fn test_min() {
    let a = array![[1, 5, 3], [2, 0, 6]];
    assert_eq!(a.min(), Some(&0));

    let a = array![[1., 5., 3.], [2., 0., 6.]];
    assert_eq!(a.min(), Some(&0.));

    let a = array![[1., 5., 3.], [2., ::std::f64::NAN, 6.]];
    assert_eq!(a.min(), None);
}

#[test]
fn test_min_skipnan() {
    let a = array![[1., 5., 3.], [2., 0., 6.]];
    assert_eq!(a.min_skipnan(), &0.);

    let a = array![[1., 5., 3.], [2., ::std::f64::NAN, 6.]];
    assert_eq!(a.min_skipnan(), &1.);
}

#[test]
fn test_min_skipnan_all_nan() {
    let a = arr2(&[[::std::f64::NAN; 3]; 2]);
    assert!(a.min_skipnan().is_nan());
}

#[test]
fn test_argmax() {
    let a = array![[1, 5, 3], [2, 0, 6]];
    assert_eq!(a.argmax(), Some((1, 2)));

    let a = array![[1., 5., 3.], [2., 0., 6.]];
    assert_eq!(a.argmax(), Some((1, 2)));

    let a = array![[1., 5., 3.], [2., ::std::f64::NAN, 6.]];
    assert_eq!(a.argmax(), None);

    let a: Array2<i32> = array![[], []];
    assert_eq!(a.argmax(), None);
}

quickcheck! {
    fn argmax_matches_max(data: Vec<f32>) -> bool {
        let a = Array1::from(data);
        a.argmax().map(|i| a[i]) == a.max().cloned()
    }
}

#[test]
fn test_max() {
    let a = array![[1, 5, 7], [2, 0, 6]];
    assert_eq!(a.max(), Some(&7));

    let a = array![[1., 5., 7.], [2., 0., 6.]];
    assert_eq!(a.max(), Some(&7.));

    let a = array![[1., 5., 7.], [2., ::std::f64::NAN, 6.]];
    assert_eq!(a.max(), None);
}

#[test]
fn test_max_skipnan() {
    let a = array![[1., 5., 7.], [2., 0., 6.]];
    assert_eq!(a.max_skipnan(), &7.);

    let a = array![[1., 5., 7.], [2., ::std::f64::NAN, 6.]];
    assert_eq!(a.max_skipnan(), &7.);
}

#[test]
fn test_max_skipnan_all_nan() {
    let a = arr2(&[[::std::f64::NAN; 3]; 2]);
    assert!(a.max_skipnan().is_nan());
}

#[test]
fn test_quantile_axis_mut_with_odd_axis_length() {
    let mut a = arr2(&[[1, 3, 2, 10], [2, 4, 3, 11], [3, 5, 6, 12]]);
    let p = a.quantile_axis_mut::<Lower>(Axis(0), n64(0.5)).unwrap();
    assert!(p == a.index_axis(Axis(0), 1));
}

#[test]
fn test_quantile_axis_mut_with_zero_axis_length() {
    let mut a = Array2::<i32>::zeros((5, 0));
    assert!(a.quantile_axis_mut::<Lower>(Axis(1), n64(0.5)).is_none());
}

#[test]
fn test_quantile_axis_mut_with_empty_array() {
    let mut a = Array2::<i32>::zeros((5, 0));
    let p = a.quantile_axis_mut::<Lower>(Axis(0), n64(0.5)).unwrap();
    assert_eq!(p.shape(), &[0]);
}

#[test]
fn test_quantile_axis_mut_with_even_axis_length() {
    let mut b = arr2(&[[1, 3, 2, 10], [2, 4, 3, 11], [3, 5, 6, 12], [4, 6, 7, 13]]);
    let q = b.quantile_axis_mut::<Lower>(Axis(0), n64(0.5)).unwrap();
    assert!(q == b.index_axis(Axis(0), 1));
}

#[test]
fn test_quantile_axis_mut_to_get_minimum() {
    let mut b = arr2(&[[1, 3, 22, 10]]);
    let q = b.quantile_axis_mut::<Lower>(Axis(1), n64(0.)).unwrap();
    assert!(q == arr1(&[1]));
}

#[test]
fn test_quantile_axis_mut_to_get_maximum() {
    let mut b = arr1(&[1, 3, 22, 10]);
    let q = b.quantile_axis_mut::<Lower>(Axis(0), n64(1.)).unwrap();
    assert!(q == arr0(22));
}

#[test]
fn test_quantile_axis_skipnan_mut_higher_opt_i32() {
    let mut a = arr2(&[[Some(4), Some(2), None, Some(1), Some(5)], [None; 5]]);
    let q = a.quantile_axis_skipnan_mut::<Higher>(Axis(1), n64(0.6)).unwrap();
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(4));
    assert!(q[1].is_none());
}

#[test]
fn test_quantile_axis_skipnan_mut_nearest_opt_i32() {
    let mut a = arr2(&[[Some(4), Some(2), None, Some(1), Some(5)], [None; 5]]);
    let q = a.quantile_axis_skipnan_mut::<Nearest>(Axis(1), n64(0.6)).unwrap();
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(4));
    assert!(q[1].is_none());
}

#[test]
fn test_quantile_axis_skipnan_mut_midpoint_opt_i32() {
    let mut a = arr2(&[[Some(4), Some(2), None, Some(1), Some(5)], [None; 5]]);
    let q = a.quantile_axis_skipnan_mut::<Midpoint>(Axis(1), n64(0.6)).unwrap();
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(3));
    assert!(q[1].is_none());
}

#[test]
fn test_quantile_axis_skipnan_mut_linear_f64() {
    let mut a = arr2(&[[1., 2., ::std::f64::NAN, 3.], [::std::f64::NAN; 4]]);
    let q = a.quantile_axis_skipnan_mut::<Linear>(Axis(1), n64(0.75)).unwrap();
    assert_eq!(q.shape(), &[2]);
    assert!((q[0] - 2.5).abs() < 1e-12);
    assert!(q[1].is_nan());
}

#[test]
fn test_quantile_axis_skipnan_mut_linear_opt_i32() {
    let mut a = arr2(&[[Some(2), Some(4), None, Some(1)], [None; 4]]);
    let q = a.quantile_axis_skipnan_mut::<Linear>(Axis(1), n64(0.75)).unwrap();
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(3));
    assert!(q[1].is_none());
}

#[test]
fn test_midpoint_overflow() {
    // Regression test
    // This triggered an overflow panic with a naive Midpoint implementation: (a+b)/2
    let mut a: Array1<u8> = array![129, 130, 130, 131];
    let median = a.quantile_mut::<Midpoint>(n64(0.5)).unwrap();
    let expected_median = 130;
    assert_eq!(median, expected_median);
}

#[quickcheck]
fn test_quantiles_mut(xs: Vec<i64>) -> bool {
    let v = Array::from_vec(xs.clone());

    // Unordered list of quantile indexes to look up, with a duplicate
    let quantile_indexes = vec![
        n64(0.75), n64(0.90), n64(0.95), n64(0.99), n64(1.),
        n64(0.), n64(0.25), n64(0.5), n64(0.5)
    ];
    let mut checks = vec![];
    checks.push(check_one_interpolation_method_for_quantiles_mut::<Linear>(v.clone(), &quantile_indexes));
    checks.push(check_one_interpolation_method_for_quantiles_mut::<Higher>(v.clone(), &quantile_indexes));
    checks.push(check_one_interpolation_method_for_quantiles_mut::<Lower>(v.clone(), &quantile_indexes));
    checks.push(check_one_interpolation_method_for_quantiles_mut::<Midpoint>(v.clone(), &quantile_indexes));
    checks.push(check_one_interpolation_method_for_quantiles_mut::<Nearest>(v.clone(), &quantile_indexes));
    checks.into_iter().all(|x| x)
}

fn check_one_interpolation_method_for_quantiles_mut<I: Interpolate<i64>>(mut v: Array1<i64>, quantile_indexes: &[N64]) -> bool
{
    let bulk_quantiles = v.quantiles_mut::<I>(&quantile_indexes);

    if v.len() == 0 {
        bulk_quantiles.is_none()
    } else {
        let bulk_quantiles = bulk_quantiles.unwrap();

        let mut checks = vec![];
        for quantile_index in quantile_indexes.iter() {
            let quantile = v.quantile_mut::<I>(*quantile_index).unwrap();
            checks.push(
                quantile == *bulk_quantiles.get(quantile_index).unwrap()
            );
        }
        checks.into_iter().all(|x| x)
    }
}

#[quickcheck]
fn test_quantiles_axis_mut(xs: Vec<u64>) -> bool {
    // We want a square matrix
    let axis_length = (xs.len() as f64).sqrt().floor() as usize;
    let xs = &xs[..axis_length.pow(2)];
    let m = Array::from_vec(xs.to_vec())
        .into_shape((axis_length, axis_length))
        .unwrap();

    // Unordered list of quantile indexes to look up, with a duplicate
    let quantile_indexes = vec![
        n64(0.75), n64(0.90), n64(0.95), n64(0.99), n64(1.),
        n64(0.), n64(0.25), n64(0.5), n64(0.5)
    ];

    // Test out all interpolation methods
    let mut checks = vec![];
    checks.push(
        check_one_interpolation_method_for_quantiles_axis_mut::<Linear>(
            m.clone(), &quantile_indexes, Axis(0)
        )
    );
    checks.push(
        check_one_interpolation_method_for_quantiles_axis_mut::<Higher>(
            m.clone(), &quantile_indexes, Axis(0)
        )
    );
    checks.push(
        check_one_interpolation_method_for_quantiles_axis_mut::<Lower>(
            m.clone(), &quantile_indexes, Axis(0)
        )
    );
    checks.push(
        check_one_interpolation_method_for_quantiles_axis_mut::<Midpoint>(
            m.clone(), &quantile_indexes, Axis(0)
        )
    );
    checks.push(
        check_one_interpolation_method_for_quantiles_axis_mut::<Nearest>(
            m.clone(), &quantile_indexes, Axis(0)
        )
    );
    checks.into_iter().all(|x| x)
}

fn check_one_interpolation_method_for_quantiles_axis_mut<I: Interpolate<u64>>(mut v: Array2<u64>, quantile_indexes: &[N64], axis: Axis) -> bool
{
    let bulk_quantiles = v.quantiles_axis_mut::<I>(axis, &quantile_indexes);

    if v.len() == 0 {
        bulk_quantiles.is_none()
    } else {
        let bulk_quantiles = bulk_quantiles.unwrap();
        let mut checks = vec![];
        for quantile_index in quantile_indexes.iter() {
            let quantile = v.quantile_axis_mut::<I>(axis, *quantile_index).unwrap();
            checks.push(
                quantile == *bulk_quantiles.get(quantile_index).unwrap()
            );
        }
        checks.into_iter().all(|x| x)
    }
}
