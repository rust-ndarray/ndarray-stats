use ndarray::prelude::*;
use ndarray::Data;
use std::hash::Hash;
use std::ops::Index;
use histogram::bins::Bin1d;

#[derive(Hash, PartialEq, Eq)]
pub struct BinNd<T: Hash + Eq> {
    projections: Vec<Bin1d<T>>,
}

impl<T> BinNd<T>
where
    T: Hash + Eq
{
    /// Creates a new instance of BinNd from a vector
    /// of its 1-dimensional projections. 
    pub fn new(projections: Vec<Bin1d<T>>) -> Self {
        if projections.is_empty() {
            panic!(
                "The 1-dimensional projections of an n-dimensional
                bin can't be empty!" 
            )
        } else {
            Self { projections }
        }
    }

    pub fn ndim(&self) -> usize {
        self.projections.len()
    }
}

/// `Bins` is a collection of non-overlapping 
/// sub-regions (`BinNd`) in a `n`-dimensional space.
pub struct BinsNd<T: Hash + Eq> {
    bins: Vec<BinNd<T>>,
    ndim: usize,
}

impl<T> BinsNd<T> 
where 
    T: Hash + Eq 
{
    /// Creates a new instance of BinNd from a vector
    /// of its 1-dimensional projections. 
    pub fn new(bins: Vec<BinNd<T>>) -> Self {
        assert!(!bins.is_empty(), "The bins collection cannot be empty!");
        // All bins must have the same number of dimensions!
        let first_bin = bins.index(0);
        let ndim = first_bin.ndim();
        &bins.iter().map(
            |b| assert_eq!(
                b.ndim(), ndim, 
                "There at least two bins with different \
                number of dimensions: {0} and {1}.", b, first_bin)
        );
        Self { bins, ndim }
    }

    /// Return `n`, the number of dimensions.
    /// 
    /// **Panics** if `bins` is empty.
    fn ndim(&self) -> usize {
        self.ndim
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
