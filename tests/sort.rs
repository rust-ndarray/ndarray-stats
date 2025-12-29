use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::{Sort1dExt, SortExt};
use quickcheck_macros::quickcheck;

#[test]
fn test_partition_mut() {
    let mut l = vec![
        arr1(&[1, 1, 1, 1, 1]),
        arr1(&[1, 3, 2, 10, 10]),
        arr1(&[2, 3, 4, 1]),
        arr1(&[
            355, 453, 452, 391, 289, 343, 44, 154, 271, 44, 314, 276, 160, 469, 191, 138, 163, 308,
            395, 3, 416, 391, 210, 354, 200,
        ]),
        arr1(&[
            84, 192, 216, 159, 89, 296, 35, 213, 456, 278, 98, 52, 308, 418, 329, 173, 286, 106,
            366, 129, 125, 450, 23, 463, 151,
        ]),
    ];
    for a in l.iter_mut() {
        let n = a.len();
        let pivot_index = n - 1;
        let pivot_value = a[pivot_index].clone();
        let partition_index = a.partition_mut(pivot_index);
        for i in 0..partition_index {
            assert!(a[i] < pivot_value);
        }
        assert_eq!(a[partition_index], pivot_value);
        for j in (partition_index + 1)..n {
            assert!(pivot_value <= a[j]);
        }
    }
}

#[test]
fn test_sorted_get_mut() {
    let a = arr1(&[1, 3, 2, 10]);
    let j = a.clone().view_mut().get_from_sorted_mut(2);
    assert_eq!(j, 3);
    let j = a.clone().view_mut().get_from_sorted_mut(1);
    assert_eq!(j, 2);
    let j = a.clone().view_mut().get_from_sorted_mut(3);
    assert_eq!(j, 10);
}

#[quickcheck]
fn test_sorted_get_many_mut(mut xs: Vec<i64>) -> bool {
    let n = xs.len();
    if n == 0 {
        true
    } else {
        let mut v = Array::from(xs.clone());

        // Insert each index twice, to get a set of indexes with duplicates, not sorted
        let mut indexes: Vec<usize> = (0..n).into_iter().collect();
        indexes.append(&mut (0..n).collect());

        let mut sorted_v = Vec::with_capacity(n);
        for (i, (key, value)) in v
            .get_many_from_sorted_mut(&Array::from(indexes))
            .into_iter()
            .enumerate()
        {
            if i != key {
                return false;
            }
            sorted_v.push(value);
        }
        xs.sort();
        println!("Sorted: {:?}. Truth: {:?}", sorted_v, xs);
        xs == sorted_v
    }
}

#[quickcheck]
fn test_sorted_get_mut_as_sorting_algorithm(mut xs: Vec<i64>) -> bool {
    let n = xs.len();
    if n == 0 {
        true
    } else {
        let mut v = Array::from(xs.clone());
        let sorted_v: Vec<_> = (0..n).map(|i| v.get_from_sorted_mut(i)).collect();
        xs.sort();
        xs == sorted_v
    }
}

#[test]
fn argsort_1d() {
    let a = array![5, 2, 0, 7, 3, 2, 8, 9];
    let correct = array![2, 1, 5, 4, 0, 3, 6, 7];
    assert_eq!(a.argsort_axis(Axis(0)), &correct);
    assert_eq!(a.argsort_axis_by(Axis(0), |a, b| a.cmp(b)), &correct);
    assert_eq!(a.argsort_axis_by_key(Axis(0), |&x| x), &correct);
}

#[test]
fn argsort_2d() {
    let a = array![[3, 5, 1, 2], [2, 0, 1, 3], [9, 4, 6, 1]];
    for (axis, correct) in [
        (Axis(0), array![[1, 1, 0, 2], [0, 2, 1, 0], [2, 0, 2, 1]]),
        (Axis(1), array![[2, 3, 0, 1], [1, 2, 0, 3], [3, 1, 2, 0]]),
    ] {
        assert_eq!(a.argsort_axis(axis), &correct);
        assert_eq!(a.argsort_axis_by(axis, |a, b| a.cmp(b)), &correct);
        assert_eq!(a.argsort_axis_by_key(axis, |&x| x), &correct);
    }
}

#[test]
fn argsort_3d() {
    let a = array![
        [[3, 5, 1, 2], [9, 7, 6, 8]],
        [[2, 0, 1, 3], [1, 2, 3, 4]],
        [[9, 4, 6, 1], [8, 5, 3, 2]],
    ];
    for (axis, correct) in [
        (
            Axis(0),
            array![
                [[1, 1, 0, 2], [1, 1, 1, 2]],
                [[0, 2, 1, 0], [2, 2, 2, 1]],
                [[2, 0, 2, 1], [0, 0, 0, 0]],
            ],
        ),
        (
            Axis(1),
            array![
                [[0, 0, 0, 0], [1, 1, 1, 1]],
                [[1, 0, 0, 0], [0, 1, 1, 1]],
                [[1, 0, 1, 0], [0, 1, 0, 1]],
            ],
        ),
        (
            Axis(2),
            array![
                [[2, 3, 0, 1], [2, 1, 3, 0]],
                [[1, 2, 0, 3], [0, 1, 2, 3]],
                [[3, 1, 2, 0], [3, 2, 1, 0]],
            ],
        ),
    ] {
        assert_eq!(a.argsort_axis(axis), &correct);
        assert_eq!(a.argsort_axis_by(axis, |a, b| a.cmp(b)), &correct);
        assert_eq!(a.argsort_axis_by_key(axis, |&x| x), &correct);
    }
}

#[test]
fn argsort_len_0_or_1_axis() {
    fn test_shape(base_shape: impl ndarray::IntoDimension) {
        let base_shape = base_shape.into_dimension();
        for ax in 0..base_shape.ndim() {
            for axis_len in [0, 1] {
                let mut shape = base_shape.clone();
                shape[ax] = axis_len;
                let a = Array::random(shape.clone(), Uniform::new(0, 100));
                let axis = Axis(ax);
                let correct = Array::zeros(shape);
                assert_eq!(a.argsort_axis(axis), &correct);
                assert_eq!(a.argsort_axis_by(axis, |a, b| a.cmp(b)), &correct);
                assert_eq!(a.argsort_axis_by_key(axis, |&x| x), &correct);
            }
        }
    }
    test_shape([1]);
    test_shape([2, 3]);
    test_shape([3, 2, 4]);
    test_shape([2, 4, 3, 2]);
}
