extern crate ndarray;
extern crate ndarray_stats;

use ndarray::prelude::*;
use ndarray_stats::{
    interpolate::{Linear, Lower},
    PercentileExt,
};

#[test]
fn test_percentile_axis_mut_with_odd_axis_length() {
    let mut a = arr2(&[[1, 3, 2, 10], [2, 4, 3, 11], [3, 5, 6, 12]]);
    let p = a.percentile_axis_mut::<Lower>(Axis(0), 0.5);
    assert!(p == a.subview(Axis(0), 1));
}

#[test]
fn test_percentile_axis_mut_with_even_axis_length() {
    let mut b = arr2(&[[1, 3, 2, 10], [2, 4, 3, 11], [3, 5, 6, 12], [4, 6, 7, 13]]);
    let q = b.percentile_axis_mut::<Lower>(Axis(0), 0.5);
    assert!(q == b.subview(Axis(0), 1));
}

#[test]
fn test_percentile_axis_mut_to_get_minimum() {
    let mut b = arr2(&[[1, 3, 22, 10]]);
    let q = b.percentile_axis_mut::<Lower>(Axis(1), 0.);
    assert!(q == arr1(&[1]));
}

#[test]
fn test_percentile_axis_mut_to_get_maximum() {
    let mut b = arr1(&[1, 3, 22, 10]);
    let q = b.percentile_axis_mut::<Lower>(Axis(0), 1.);
    assert!(q == arr0(22));
}

// TODO: See https://github.com/SergiusIW/noisy_float-rs/pull/19
// #[test]
// fn test_percentile_axis_skipnan_mut_f64() {
//     let mut a = arr2(&[[1., 2., ::std::f64::NAN, 3.], [::std::f64::NAN; 4]]);
//     let q = a.percentile_axis_skipnan_mut::<Linear>(Axis(1), 0.75);
//     assert_eq!(q.shape(), &[2]);
//     assert!((q[0] - 2.5).abs() < 1e-12);
//     assert!(q[1].is_nan());
// }

#[test]
fn test_percentile_axis_skipnan_mut_opt_i32() {
    let mut a = arr2(&[[Some(1), Some(2), None, Some(4)], [None; 4]]);
    let q = a.percentile_axis_skipnan_mut::<Linear>(Axis(1), 0.75);
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(3));
    assert!(q[1].is_none());
}
