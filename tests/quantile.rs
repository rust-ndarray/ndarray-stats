#[macro_use(array)]
extern crate ndarray;
extern crate ndarray_stats;

use ndarray::prelude::*;
use ndarray_stats::{
    interpolate::{Higher, Linear, Lower, Midpoint, Nearest},
    QuantileExt,
};

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
    let p = a.quantile_axis_mut::<Lower>(Axis(0), 0.5);
    assert!(p == a.index_axis(Axis(0), 1));
}

#[test]
#[should_panic]
fn test_quantile_axis_mut_with_zero_axis_length() {
    let mut a = Array2::<i32>::zeros((5, 0));
    a.quantile_axis_mut::<Lower>(Axis(1), 0.5);
}

#[test]
fn test_quantile_axis_mut_with_empty_array() {
    let mut a = Array2::<i32>::zeros((5, 0));
    let p = a.quantile_axis_mut::<Lower>(Axis(0), 0.5);
    assert_eq!(p.shape(), &[0]);
}

#[test]
fn test_quantile_axis_mut_with_even_axis_length() {
    let mut b = arr2(&[[1, 3, 2, 10], [2, 4, 3, 11], [3, 5, 6, 12], [4, 6, 7, 13]]);
    let q = b.quantile_axis_mut::<Lower>(Axis(0), 0.5);
    assert!(q == b.index_axis(Axis(0), 1));
}

#[test]
fn test_quantile_axis_mut_to_get_minimum() {
    let mut b = arr2(&[[1, 3, 22, 10]]);
    let q = b.quantile_axis_mut::<Lower>(Axis(1), 0.);
    assert!(q == arr1(&[1]));
}

#[test]
fn test_quantile_axis_mut_to_get_maximum() {
    let mut b = arr1(&[1, 3, 22, 10]);
    let q = b.quantile_axis_mut::<Lower>(Axis(0), 1.);
    assert!(q == arr0(22));
}

#[test]
fn test_quantile_axis_skipnan_mut_higher_opt_i32() {
    let mut a = arr2(&[[Some(4), Some(2), None, Some(1), Some(5)], [None; 5]]);
    let q = a.quantile_axis_skipnan_mut::<Higher>(Axis(1), 0.6);
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(4));
    assert!(q[1].is_none());
}

#[test]
fn test_quantile_axis_skipnan_mut_nearest_opt_i32() {
    let mut a = arr2(&[[Some(4), Some(2), None, Some(1), Some(5)], [None; 5]]);
    let q = a.quantile_axis_skipnan_mut::<Nearest>(Axis(1), 0.6);
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(4));
    assert!(q[1].is_none());
}

#[test]
fn test_quantile_axis_skipnan_mut_midpoint_opt_i32() {
    let mut a = arr2(&[[Some(4), Some(2), None, Some(1), Some(5)], [None; 5]]);
    let q = a.quantile_axis_skipnan_mut::<Midpoint>(Axis(1), 0.6);
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(3));
    assert!(q[1].is_none());
}

#[test]
fn test_quantile_axis_skipnan_mut_linear_f64() {
    let mut a = arr2(&[[1., 2., ::std::f64::NAN, 3.], [::std::f64::NAN; 4]]);
    let q = a.quantile_axis_skipnan_mut::<Linear>(Axis(1), 0.75);
    assert_eq!(q.shape(), &[2]);
    assert!((q[0] - 2.5).abs() < 1e-12);
    assert!(q[1].is_nan());
}

#[test]
fn test_quantile_axis_skipnan_mut_linear_opt_i32() {
    let mut a = arr2(&[[Some(2), Some(4), None, Some(1)], [None; 4]]);
    let q = a.quantile_axis_skipnan_mut::<Linear>(Axis(1), 0.75);
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(3));
    assert!(q[1].is_none());
}
