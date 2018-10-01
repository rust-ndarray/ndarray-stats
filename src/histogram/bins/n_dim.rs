use ndarray::prelude::*;
use ndarray::Data;
use std::hash::Hash;
use histogram::bins::Bin1d;

#[derive(Hash, PartialEq, Eq)]
pub struct BinNd<T: Hash + Eq> {
    projections: Vec<Bin1d<T>>,
}

/// `Bins` is a collection of non-overlapping 
/// sub-regions (`BinNd`) in a `n`-dimensional space.
pub struct BinsNd<T: Hash + Eq> {
    bins: Vec<BinNd<T>>,
}


impl<T> BinsNd<T> 
where 
    T: Hash + Eq 
{
    /// Return `n`, the number of dimensions.
    fn ndim(&self) -> usize {
        unimplemented!() 
    }

    /// Given a point `P`, it returns an `Option`:
    /// - `Some(B)`, if `P` belongs to the `Bin` `B`;
    /// - `None`, if `P` does not belong to any `Bin` in `Bins`.
    /// 
    /// **Panics** if `P.ndim()` is different from `Bins.ndim()`. 
    pub fn find<S, D>(&self, _point: ArrayBase<S, D>) -> Option<BinNd<T>>
    where
        S: Data<Elem = T>,
        D: Dimension,
    {
        unimplemented!()
    }
}
