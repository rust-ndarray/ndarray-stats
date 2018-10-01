use ndarray::prelude::*;
use ndarray::Data;
use std::hash::Hash;
use std::ops::Index;
use std::fmt;
use histogram::bins::Bin1d;

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct BinNd<T: Hash + Eq + fmt::Debug + Clone> {
    projections: Vec<Bin1d<T>>,
}

impl<T> fmt::Display for BinNd<T>
where
    T: Hash + Eq + fmt::Debug + Clone
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let repr = self.projections.iter().map(
            |p| format!("{}", p)
        ).collect::<Vec<String>>().join("x");
        write!(f, "{}", repr)
    }
}

impl<T> BinNd<T>
where
    T: Hash + Eq + fmt::Debug + Clone + PartialOrd
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

    pub fn contains(&self, point: ArrayView1<T>) -> bool
    {
        point.iter().
            zip(self.projections.iter()).
            map(|(element, projection)| projection.contains(element)).
            fold(true, |acc, v| acc & v)
    }
}

/// `Bins` is a collection of non-overlapping
/// sub-regions (`BinNd`) in a `n`-dimensional space.
pub struct BinsNd<T: Hash + Eq + fmt::Debug + Clone> {
    bins: Vec<BinNd<T>>,
    ndim: usize,
}

impl<T> BinsNd<T>
where
    T: Hash + Eq + fmt::Debug + Clone + PartialOrd
{
    /// Creates a new instance of BinNd from a vector
    /// of its 1-dimensional projections.
    pub fn new(bins: Vec<BinNd<T>>) -> Self {
        assert!(!bins.is_empty(), "The bins collection cannot be empty!");
        // All bins must have the same number of dimensions!
        let ndim = {
            let first_bin = bins.index(0);
            let ndim = first_bin.ndim();
            &bins.iter().map(
                |b| assert_eq!(
                    b.ndim(), ndim,
                    "There at least two bins with different \
                    number of dimensions: {0} and {1}.", b, first_bin)
            );
            ndim
        };
        Self { bins, ndim }
    }

    /// Return `n`, the number of dimensions.
    ///
    /// **Panics** if `bins` is empty.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Given a point `P`, it returns an `Option`:
    /// - `Some(B)`, if `P` belongs to the `Bin` `B`;
    /// - `None`, if `P` does not belong to any `Bin` in `Bins`.
    ///
    /// **Panics** if `P.ndim()` is different from `Bins.ndim()`.
    pub fn find<S>(&self, point: ArrayBase<S, Ix1>) -> Option<BinNd<T>>
    where
        S: Data<Elem = T>,
    {
        for bin in self.bins.iter() {
            if bin.contains(point.view()) {
                return Some((*bin).clone())
            }
        }
        None
    }
}