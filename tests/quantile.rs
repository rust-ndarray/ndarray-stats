use itertools::izip;
use ndarray::array;
use ndarray::prelude::*;
use ndarray_stats::{
    errors::{EmptyInput, MinMaxError, QuantileError},
    interpolate::{Higher, Interpolate, Linear, Lower, Midpoint, Nearest},
    Quantile1dExt, QuantileExt,
};
use noisy_float::types::{n64, N64};
use quickcheck_macros::quickcheck;

#[test]
fn test_argmin() {
    let a = array![[1, 5, 3], [2, 0, 6]];
    assert_eq!(a.argmin(), Ok((1, 1)));

    let a = array![[1., 5., 3.], [2., 0., 6.]];
    assert_eq!(a.argmin(), Ok((1, 1)));

    let a = array![[1., 5., 3.], [2., ::std::f64::NAN, 6.]];
    assert_eq!(a.argmin(), Err(MinMaxError::UndefinedOrder));

    let a: Array2<i32> = array![[], []];
    assert_eq!(a.argmin(), Err(MinMaxError::EmptyInput));
}

#[quickcheck]
fn argmin_matches_min(data: Vec<f32>) -> bool {
    let a = Array1::from(data);
    a.argmin().map(|i| &a[i]) == a.min()
}

#[test]
fn test_argmin_skipnan() {
    let a = array![[1., 5., 3.], [2., 0., 6.]];
    assert_eq!(a.argmin_skipnan(), Ok((1, 1)));

    let a = array![[1., 5., 3.], [2., ::std::f64::NAN, 6.]];
    assert_eq!(a.argmin_skipnan(), Ok((0, 0)));

    let a = array![[::std::f64::NAN, 5., 3.], [2., ::std::f64::NAN, 6.]];
    assert_eq!(a.argmin_skipnan(), Ok((1, 0)));

    let a: Array2<f64> = array![[], []];
    assert_eq!(a.argmin_skipnan(), Err(EmptyInput));

    let a = arr2(&[[::std::f64::NAN; 2]; 2]);
    assert_eq!(a.argmin_skipnan(), Err(EmptyInput));
}

#[quickcheck]
fn argmin_skipnan_matches_min_skipnan(data: Vec<Option<i32>>) -> bool {
    let a = Array1::from(data);
    let min = a.min_skipnan();
    let argmin = a.argmin_skipnan();
    if min.is_none() {
        argmin == Err(EmptyInput)
    } else {
        a[argmin.unwrap()] == *min
    }
}

#[test]
fn test_min() {
    let a = array![[1, 5, 3], [2, 0, 6]];
    assert_eq!(a.min(), Ok(&0));

    let a = array![[1., 5., 3.], [2., 0., 6.]];
    assert_eq!(a.min(), Ok(&0.));

    let a = array![[1., 5., 3.], [2., ::std::f64::NAN, 6.]];
    assert_eq!(a.min(), Err(MinMaxError::UndefinedOrder));
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
    assert_eq!(a.argmax(), Ok((1, 2)));

    let a = array![[1., 5., 3.], [2., 0., 6.]];
    assert_eq!(a.argmax(), Ok((1, 2)));

    let a = array![[1., 5., 3.], [2., ::std::f64::NAN, 6.]];
    assert_eq!(a.argmax(), Err(MinMaxError::UndefinedOrder));

    let a: Array2<i32> = array![[], []];
    assert_eq!(a.argmax(), Err(MinMaxError::EmptyInput));
}

#[quickcheck]
fn argmax_matches_max(data: Vec<f32>) -> bool {
    let a = Array1::from(data);
    a.argmax().map(|i| &a[i]) == a.max()
}

#[test]
fn test_argmax_skipnan() {
    let a = array![[1., 5., 3.], [2., 0., 6.]];
    assert_eq!(a.argmax_skipnan(), Ok((1, 2)));

    let a = array![[1., 5., 3.], [2., ::std::f64::NAN, ::std::f64::NAN]];
    assert_eq!(a.argmax_skipnan(), Ok((0, 1)));

    let a = array![
        [::std::f64::NAN, ::std::f64::NAN, 3.],
        [2., ::std::f64::NAN, 6.]
    ];
    assert_eq!(a.argmax_skipnan(), Ok((1, 2)));

    let a: Array2<f64> = array![[], []];
    assert_eq!(a.argmax_skipnan(), Err(EmptyInput));

    let a = arr2(&[[::std::f64::NAN; 2]; 2]);
    assert_eq!(a.argmax_skipnan(), Err(EmptyInput));
}

#[quickcheck]
fn argmax_skipnan_matches_max_skipnan(data: Vec<Option<i32>>) -> bool {
    let a = Array1::from(data);
    let max = a.max_skipnan();
    let argmax = a.argmax_skipnan();
    if max.is_none() {
        argmax == Err(EmptyInput)
    } else {
        a[argmax.unwrap()] == *max
    }
}

#[test]
fn test_max() {
    let a = array![[1, 5, 7], [2, 0, 6]];
    assert_eq!(a.max(), Ok(&7));

    let a = array![[1., 5., 7.], [2., 0., 6.]];
    assert_eq!(a.max(), Ok(&7.));

    let a = array![[1., 5., 7.], [2., ::std::f64::NAN, 6.]];
    assert_eq!(a.max(), Err(MinMaxError::UndefinedOrder));
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
    let p = a.quantile_axis_mut(Axis(0), n64(0.5), &Lower).unwrap();
    assert!(p == a.index_axis(Axis(0), 1));
}

#[test]
fn test_quantile_axis_mut_with_zero_axis_length() {
    let mut a = Array2::<i32>::zeros((5, 0));
    assert_eq!(
        a.quantile_axis_mut(Axis(1), n64(0.5), &Lower),
        Err(QuantileError::EmptyInput)
    );
}

#[test]
fn test_quantile_axis_mut_with_empty_array() {
    let mut a = Array2::<i32>::zeros((5, 0));
    let p = a.quantile_axis_mut(Axis(0), n64(0.5), &Lower).unwrap();
    assert_eq!(p.shape(), &[0]);
}

#[test]
fn test_quantile_axis_mut_with_even_axis_length() {
    let mut b = arr2(&[[1, 3, 2, 10], [2, 4, 3, 11], [3, 5, 6, 12], [4, 6, 7, 13]]);
    let q = b.quantile_axis_mut(Axis(0), n64(0.5), &Lower).unwrap();
    assert!(q == b.index_axis(Axis(0), 1));
}

#[test]
fn test_quantile_axis_mut_to_get_minimum() {
    let mut b = arr2(&[[1, 3, 22, 10]]);
    let q = b.quantile_axis_mut(Axis(1), n64(0.), &Lower).unwrap();
    assert!(q == arr1(&[1]));
}

#[test]
fn test_quantile_axis_mut_to_get_maximum() {
    let mut b = arr1(&[1, 3, 22, 10]);
    let q = b.quantile_axis_mut(Axis(0), n64(1.), &Lower).unwrap();
    assert!(q == arr0(22));
}

#[test]
fn test_quantile_axis_skipnan_mut_higher_opt_i32() {
    let mut a = arr2(&[[Some(4), Some(2), None, Some(1), Some(5)], [None; 5]]);
    let q = a
        .quantile_axis_skipnan_mut(Axis(1), n64(0.6), &Higher)
        .unwrap();
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(4));
    assert!(q[1].is_none());
}

#[test]
fn test_quantile_axis_skipnan_mut_nearest_opt_i32() {
    let mut a = arr2(&[[Some(4), Some(2), None, Some(1), Some(5)], [None; 5]]);
    let q = a
        .quantile_axis_skipnan_mut(Axis(1), n64(0.6), &Nearest)
        .unwrap();
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(4));
    assert!(q[1].is_none());
}

#[test]
fn test_quantile_axis_skipnan_mut_midpoint_opt_i32() {
    let mut a = arr2(&[[Some(4), Some(2), None, Some(1), Some(5)], [None; 5]]);
    let q = a
        .quantile_axis_skipnan_mut(Axis(1), n64(0.6), &Midpoint)
        .unwrap();
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(3));
    assert!(q[1].is_none());
}

#[test]
fn test_quantile_axis_skipnan_mut_linear_f64() {
    let mut a = arr2(&[[1., 2., ::std::f64::NAN, 3.], [::std::f64::NAN; 4]]);
    let q = a
        .quantile_axis_skipnan_mut(Axis(1), n64(0.75), &Linear)
        .unwrap();
    assert_eq!(q.shape(), &[2]);
    assert!((q[0] - 2.5).abs() < 1e-12);
    assert!(q[1].is_nan());
}

#[test]
fn test_quantile_axis_skipnan_mut_linear_opt_i32() {
    let mut a = arr2(&[[Some(2), Some(4), None, Some(1)], [None; 4]]);
    let q = a
        .quantile_axis_skipnan_mut(Axis(1), n64(0.75), &Linear)
        .unwrap();
    assert_eq!(q.shape(), &[2]);
    assert_eq!(q[0], Some(3));
    assert!(q[1].is_none());
}

#[test]
fn test_midpoint_overflow() {
    // Regression test
    // This triggered an overflow panic with a naive Midpoint implementation: (a+b)/2
    let mut a: Array1<u8> = array![129, 130, 130, 131];
    let median = a.quantile_mut(n64(0.5), &Midpoint).unwrap();
    let expected_median = 130;
    assert_eq!(median, expected_median);
}

#[quickcheck]
fn test_quantiles_mut(xs: Vec<i64>) -> bool {
    let v = Array::from(xs.clone());

    // Unordered list of quantile indexes to look up, with a duplicate
    let quantile_indexes = Array::from(vec![
        n64(0.75),
        n64(0.90),
        n64(0.95),
        n64(0.99),
        n64(1.),
        n64(0.),
        n64(0.25),
        n64(0.5),
        n64(0.5),
    ]);
    let mut correct = true;
    correct &= check_one_interpolation_method_for_quantiles_mut(
        v.clone(),
        quantile_indexes.view(),
        &Linear,
    );
    correct &= check_one_interpolation_method_for_quantiles_mut(
        v.clone(),
        quantile_indexes.view(),
        &Higher,
    );
    correct &= check_one_interpolation_method_for_quantiles_mut(
        v.clone(),
        quantile_indexes.view(),
        &Lower,
    );
    correct &= check_one_interpolation_method_for_quantiles_mut(
        v.clone(),
        quantile_indexes.view(),
        &Midpoint,
    );
    correct &= check_one_interpolation_method_for_quantiles_mut(
        v.clone(),
        quantile_indexes.view(),
        &Nearest,
    );
    correct
}

fn check_one_interpolation_method_for_quantiles_mut(
    mut v: Array1<i64>,
    quantile_indexes: ArrayView1<'_, N64>,
    interpolate: &impl Interpolate<i64>,
) -> bool {
    let bulk_quantiles = v.clone().quantiles_mut(&quantile_indexes, interpolate);

    if v.len() == 0 {
        bulk_quantiles.is_err()
    } else {
        let bulk_quantiles = bulk_quantiles.unwrap();
        izip!(quantile_indexes, &bulk_quantiles).all(|(&quantile_index, &quantile)| {
            quantile == v.quantile_mut(quantile_index, interpolate).unwrap()
        })
    }
}

#[quickcheck]
fn test_quantiles_axis_mut(mut xs: Vec<u64>) -> bool {
    // We want a square matrix
    let axis_length = (xs.len() as f64).sqrt().floor() as usize;
    xs.truncate(axis_length * axis_length);
    let m = Array::from_shape_vec((axis_length, axis_length), xs).unwrap();

    // Unordered list of quantile indexes to look up, with a duplicate
    let quantile_indexes = Array::from(vec![
        n64(0.75),
        n64(0.90),
        n64(0.95),
        n64(0.99),
        n64(1.),
        n64(0.),
        n64(0.25),
        n64(0.5),
        n64(0.5),
    ]);

    // Test out all interpolation methods
    let mut correct = true;
    correct &= check_one_interpolation_method_for_quantiles_axis_mut(
        m.clone(),
        quantile_indexes.view(),
        Axis(0),
        &Linear,
    );
    correct &= check_one_interpolation_method_for_quantiles_axis_mut(
        m.clone(),
        quantile_indexes.view(),
        Axis(0),
        &Higher,
    );
    correct &= check_one_interpolation_method_for_quantiles_axis_mut(
        m.clone(),
        quantile_indexes.view(),
        Axis(0),
        &Lower,
    );
    correct &= check_one_interpolation_method_for_quantiles_axis_mut(
        m.clone(),
        quantile_indexes.view(),
        Axis(0),
        &Midpoint,
    );
    correct &= check_one_interpolation_method_for_quantiles_axis_mut(
        m.clone(),
        quantile_indexes.view(),
        Axis(0),
        &Nearest,
    );
    correct
}

fn check_one_interpolation_method_for_quantiles_axis_mut(
    mut v: Array2<u64>,
    quantile_indexes: ArrayView1<'_, N64>,
    axis: Axis,
    interpolate: &impl Interpolate<u64>,
) -> bool {
    let bulk_quantiles = v
        .clone()
        .quantiles_axis_mut(axis, &quantile_indexes, interpolate);

    if v.len() == 0 {
        bulk_quantiles.is_err()
    } else {
        let bulk_quantiles = bulk_quantiles.unwrap();
        izip!(quantile_indexes, bulk_quantiles.axis_iter(axis)).all(
            |(&quantile_index, quantile)| {
                quantile
                    == v.quantile_axis_mut(axis, quantile_index, interpolate)
                        .unwrap()
            },
        )
    }
}
