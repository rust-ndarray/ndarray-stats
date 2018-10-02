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
    /// Creates a new instance of BinNd from the ordered sequence
    /// of its 1-dimensional projections on the coordinate axes.
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
    T: PartialOrd
{
    /// Return `true` if `point` is in the bin, `false` otherwise.
    ///
    /// **Panics** if `point`'s dimensionality
    /// (`point.into_iter().len()`) is different from `self.ndim()`.
    ///
    /// # Example
    ///
    /// ```
    /// extern crate ndarray_stats;
    /// extern crate noisy_float;
    /// use noisy_float::types::n64;
    /// use ndarray_stats::{BinNd, Bin1d};
    ///
    /// let projections = vec![
    ///     Bin1d::RangeFrom(n64(0.)..),
    ///     Bin1d::RangeFrom(n64(0.)..),
    /// ];
    /// let first_quadrant = BinNd::new(projections);
    /// let good_point = vec![n64(1e6), n64(1e8)];
    /// let bad_point = vec![n64(-1.), n64(0.)];
    /// assert!(first_quadrant.contains(&good_point));
    /// assert!(!first_quadrant.contains(&bad_point));
    /// ```
    pub fn contains<S, I>(&self, point: S) -> bool
    where
        S: IntoIterator<Item=&'a T, IntoIter=I>,
        I: Iterator<Item=&'a T> + ExactSizeIterator,
    {
        let point_iter = point.into_iter();
        assert_eq!(point_iter.len(), self.ndim());
        point_iter.zip(self.projections.iter()).
            map(|(element, projection)| projection.contains(element)).
            fold(true, |acc, v| acc & v)
    }
}

/// `Bins` is a collection of non-overlapping
/// sub-regions (`BinNd`) in an `n`-dimensional space.
#[derive(Clone, Debug)]
pub struct BinsNd<T> {
    bins: Vec<BinNd<T>>,
    ndim: usize,
}

impl<T> BinsNd<T>
where
    T: fmt::Debug
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
}

impl<T> BinsNd<T>
{
    /// Return `n`, the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }
}

impl<T> BinsNd<T>
where
    T: PartialOrd + Clone
{
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
            if bin.contains(point.view()){
                return Some((*bin).clone())
            }
        }
        None
    }
}