use ndarray::prelude::*;
use ndarray::Data;
use std::hash::Hash;

#[derive(Hash, PartialEq, Eq)]
pub struct Bin1d<T: Hash + Eq> {
    left: T,
    right: T,
}

/// `Bins` is a collection of non-overlapping 
/// intervals (`Bin1d`) in a 1-dimensional space.
pub struct Bins1d<T: Hash + Eq> {
    bins: Vec<Bin1d<T>>,
}

impl<T> Bins1d<T> 
where 
    T: Hash + Eq 
{
    /// Given a point `P`, it returns an `Option`:
    /// - `Some(B)`, if `P` belongs to the `Bin` `B`;
    /// - `None`, if `P` does not belong to any `Bin` in `Bins`.
    /// 
    /// **Panics** if `P.ndim()` is different from `Bins.ndim()`. 
    pub fn find<S, D>(&self, _point: ArrayBase<S, D>) -> Option<Bin1d<T>>
    where
        S: Data<Elem = T>,
        D: Dimension,
    {
        unimplemented!()
    }
}