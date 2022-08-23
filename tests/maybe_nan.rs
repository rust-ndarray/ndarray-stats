use ndarray::prelude::*;
use ndarray_stats::MaybeNan;
use noisy_float::types::{n64, N64};

#[test]
fn remove_nan_mut_nonstandard_layout() {
    fn eq_unordered(mut a: Vec<N64>, mut b: Vec<N64>) -> bool {
        a.sort();
        b.sort();
        a == b
    }
    let a = aview1(&[1., 2., f64::NAN, f64::NAN, 3., f64::NAN, 4., 5.]);
    {
        let mut a = a.to_owned();
        let v = f64::remove_nan_mut(a.slice_mut(s![..;2]));
        assert!(eq_unordered(v.to_vec(), vec![n64(1.), n64(3.), n64(4.)]));
    }
    {
        let mut a = a.to_owned();
        let v = f64::remove_nan_mut(a.slice_mut(s![..;-1]));
        assert!(eq_unordered(
            v.to_vec(),
            vec![n64(5.), n64(4.), n64(3.), n64(2.), n64(1.)],
        ));
    }
    {
        let mut a = a.to_owned();
        let v = f64::remove_nan_mut(a.slice_mut(s![..;-2]));
        assert!(eq_unordered(v.to_vec(), vec![n64(5.), n64(2.)]));
    }
}
