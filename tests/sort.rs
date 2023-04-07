use ndarray::prelude::*;
use ndarray_slice::Slice1Ext;
use quickcheck_macros::quickcheck;

#[test]
fn test_sorted_get_mut() {
    let a = arr1(&[1, 3, 2, 10]);
    let j = *a.clone().view_mut().select_nth_unstable(2).1;
    assert_eq!(j, 3);
    let j = *a.clone().view_mut().select_nth_unstable(1).1;
    assert_eq!(j, 2);
    let j = *a.clone().view_mut().select_nth_unstable(3).1;
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
        let mut indexes = Array::from(indexes);
        indexes.sort_unstable();
        let (indexes, _duplicates) = indexes.partition_dedup();

        let mut sorted_v = Vec::with_capacity(n);
        for value in v
            .select_many_nth_unstable(&indexes)
            .0
            .into_iter()
            .map(|(_, value)| *value)
        {
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
        let sorted_v: Vec<_> = (0..n).map(|i| *v.select_nth_unstable(i).1).collect();
        xs.sort();
        xs == sorted_v
    }
}
