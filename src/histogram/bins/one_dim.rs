use ndarray::prelude::*;
use ndarray::Data;
use std::fmt;
use std::hash::Hash;
use std::cmp::Ordering;

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct Bin1d<T: Hash + Eq> {
    left: T,
    right: T,
}

impl<T> Bin1d<T>
where
    T: Hash + Eq + fmt::Debug + Clone + PartialOrd
{
    pub fn contains(&self, point: &T) -> bool
    {
        match point.partial_cmp(&self.left) {
            Some(Ordering::Greater) => {
                match point.partial_cmp(&self.right) {
                    Some(Ordering::Less) => true,
                    _ => false
                }
            },
            _ => false
        }
    }
}

impl<T> fmt::Display for Bin1d<T>
where
    T: Hash + Eq + fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({0:?}, {1:?})", self.left, self.right)
    }
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