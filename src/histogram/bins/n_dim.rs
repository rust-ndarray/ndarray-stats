use ndarray::prelude::*;
use ndarray::Data;
use std::ops::Index;
use std::fmt;
use histogram::bins::Bin1d;

/// `n`-dimensional bin: `I_1xI_2x..xI_n` where
/// `I_k` is a one-dimensional interval (`Bin1d`).
///
/// It is instantiated by specifying the ordered sequence
/// of its 1-dimensional projections on the coordinate axes.
///
/// # Example
///
/// ```
/// #[macro_use(array)]
/// extern crate ndarray;
/// extern crate ndarray_stats;
/// extern crate noisy_float;
/// use noisy_float::types::n64;
/// use ndarray_stats::{BinNd, Bin1d};
///
/// # fn main() {
/// let projections = vec![
///     Bin1d::RangeInclusive(n64(0.)..=n64(1.)),
///     Bin1d::RangeInclusive(n64(0.)..=n64(1.)),
/// ];
/// let unit_square = BinNd::new(projections);
/// let point = array![n64(0.5), n64(0.5)];
/// assert!(unit_square.contains(point.view()));
/// # }
/// ```
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct BinNd<T> {
    projections: Vec<Bin1d<T>>,
}

impl<T> fmt::Display for BinNd<T>
where
    T: fmt::Debug
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
    T: fmt::Debug
{
    /// Creates a new instance of `BinNd` from the ordered sequence
    /// of its 1-dimensional projections on the coordinate axes.
    ///
    /// **Panics** if `projections` is empty.
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
}

impl<T> BinNd<T>
{
    /// Return `n`, the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.projections.len()
    }
}

impl<'a, T: 'a> BinNd<T>
where
    T: PartialOrd + fmt::Debug,
{
    /// Return `true` if `point` is in the bin, `false` otherwise.
    ///
    /// **Panics** if `point`'s dimensionality
    /// (`point.len()`) is different from `self.ndim()`.
    ///
    /// # Example
    ///
    /// ```
    /// #[macro_use(array)]
    /// extern crate ndarray;
    /// extern crate ndarray_stats;
    /// extern crate noisy_float;
    /// use noisy_float::types::n64;
    /// use ndarray_stats::{BinNd, Bin1d};
    ///
    /// # fn main() {
    /// let projections = vec![
    ///     Bin1d::RangeFrom(n64(0.)..),
    ///     Bin1d::RangeFrom(n64(0.)..),
    /// ];
    /// let first_quadrant = BinNd::new(projections);
    /// let good_point = array![n64(1e6), n64(1e8)];
    /// let bad_point = array![n64(-1.), n64(0.)];
    /// assert!(first_quadrant.contains(good_point.view()));
    /// assert!(!first_quadrant.contains(bad_point.view()));
    /// # }
    /// ```
    pub fn contains<S>(&self, point: ArrayBase<S, Ix1>) -> bool
    where
        S: Data<Elem=T>,
    {
        assert_eq!(point.len(), self.ndim(),
            "Dimensionalities do not match. Point {0:?} has {1} dimensions. \
             Bin {2:?} has {3} dimensions", point, point.len(), self, self.ndim());
        point.iter().zip(self.projections.iter()).
            map(|(element, projection)| projection.contains(element)).
            fold(true, |acc, v| acc & v)
    }
}

/// `BinsNd` is a collection of sub-regions (`BinNd`)
/// in an `n`-dimensional space.
///
/// It is not required (or enforced) that the sub-regions
/// in `self` must be not-overlapping.
#[derive(Clone, Debug)]
pub struct BinsNd<T> {
    bins: Vec<BinNd<T>>,
    ndim: usize,
}

impl<T> BinsNd<T>
{
    /// Creates a new instance of `BinsNd` from a vector
    /// of `BinNd`.
    ///
    /// **Panics** if `bins` is empty or if there are two bins in `bins`
    /// with different dimensionality.
    pub fn new(bins: Vec<BinNd<T>>) -> Self {
        assert!(!bins.is_empty(), "The bins collection cannot be empty!");
        // All bins must have the same number of dimensions!
        let ndim = {
            let ndim = bins.index(0).ndim();
            let flag = &bins.iter().
                map(|b| b.ndim() == ndim).
                fold(true, |acc, new| acc & new);
            // It would be better to print the bad bins as well!
            assert!(flag, "There at least two bins with different \
                           number of dimensions");
            ndim
        };
        Self { bins, ndim }
    }

    /// Return `n`, the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }
}

impl<'a, T: 'a> BinsNd<T>
where
    T: PartialOrd + Clone + fmt::Debug,
{
    /// Given a point `P`, it returns an `Option`:
    /// - `Some(B)`, if `P` belongs to the `Bin` `B`;
    /// - `None`, if `P` does not belong to any `Bin` in `self`.
    ///
    /// If more than one bin in `self` contains `P`, no assumptions
    /// can be made on which bin will be returned by `find`.
    ///
    /// **Panics** if `P.ndim()` is different from `Bins.ndim()`.
    pub fn find<S>(&self, point: ArrayBase<S, Ix1>) -> Option<BinNd<T>>
    where
        S: Data<Elem=T>,
    {
        for bin in self.bins.iter() {
            if bin.contains(point.view()){
                return Some((*bin).clone())
            }
        }
        None
    }
}

#[cfg(test)]
mod bin_nd_tests {
    use super::*;

    #[test]
    #[should_panic]
    fn new_w_empty_vec() {
        let _: BinNd<i32> = BinNd::new(vec![]);
    }

    #[test]
    #[should_panic]
    fn contains_w_dimensions_mismatch() {
        let bin2d = BinNd::new(
            vec![Bin1d::Range(0..1), Bin1d::Range(2..4)]
        );
        let point1d = array![1];
        bin2d.contains(point1d.view());
    }

    #[test]
    fn contains_w_matching_dimensions() {
        let bin2d = BinNd::new(
            vec![Bin1d::Range(0..1), Bin1d::Range(2..4)]
        );
        let point2d = array![0, 3];
        bin2d.contains(point2d.view());
    }
}

mod bins_nd_tests {
    use super::*;

    #[test]
    #[should_panic]
    fn new_w_empty_vec() {
        let _: BinsNd<i32> = BinsNd::new(vec![]);
    }

    #[test]
    #[should_panic]
    fn new_w_bins_of_different_dimensions() {
        let bin1d = BinNd::new(vec![Bin1d::Range(0..5)]);
        let bin2d = BinNd::new(vec![Bin1d::Range(0..5),
                                    Bin1d::RangeFrom(2..)]);
        assert_eq!(bin1d.ndim(), 1);
        assert_eq!(bin2d.ndim(), 2);
        let _: BinsNd<i32> = BinsNd::new(
            vec![bin1d, bin2d],
        );
    }

    #[test]
    #[should_panic]
    fn find_w_mismatched_dimensions() {
        let bin2d = BinNd::new(vec![Bin1d::Range(0..5),
                                    Bin1d::RangeFrom(2..)]);
        let point3d = array![1, 2, 3];
        let bins = BinsNd::new(vec![bin2d]);
        bins.find(point3d);
    }
}