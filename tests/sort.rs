extern crate ndarray;
extern crate ndarray_stats;

use ndarray::prelude::*;
use ndarray_stats::Sort1dExt;

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
        assert!(a[partition_index] == pivot_value);
        for j in (partition_index + 1)..n {
            assert!(pivot_value <= a[j]);
        }
    }
}

#[test]
fn test_sorted_get_mut() {
    let a = arr1(&[1, 3, 2, 10]);
    let j = a.clone().view_mut().sorted_get_mut(2);
    assert_eq!(j, 3);
    let j = a.clone().view_mut().sorted_get_mut(1);
    assert_eq!(j, 2);
    let j = a.clone().view_mut().sorted_get_mut(3);
    assert_eq!(j, 10);
}
