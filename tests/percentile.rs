extern crate ndarray;
extern crate ndarray_stats;

use ndarray::prelude::*;
use ndarray_stats::{interpolate::Lower, PercentileExt};

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
