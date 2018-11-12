//! Strategies to build [`Bins`]s and [`Grid`]s (using [`GridBuilder`]) inferring
//! optimal parameters directly from data.
//!
//! The docs for each strategy have been taken almost verbatim from [`NumPy`].
//!
//! [`Bins`]: ../struct.Bins.html
//! [`Grid`]: ../struct.Grid.html
//! [`GridBuilder`]: ../struct.GridBuilder.html
//! [`NumPy`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
use ndarray::prelude::*;
use ndarray::Data;
use num_traits::{FromPrimitive, NumOps};
use super::super::{QuantileExt, QuantileExt1d};
use super::super::interpolate::Nearest;
use super::{Edges, Bins};


/// A trait implemented by all strategies to build [`Bins`]
/// with parameters inferred from observations.
///
/// A `BinsBuildingStrategy` is required by [`GridBuilder`]
/// to know how to build a [`Grid`]'s projections on the
/// coordinate axes.
///
/// [`Bins`]: ../struct.Bins.html
/// [`Grid`]: ../struct.Grid.html
/// [`GridBuilder`]: ../struct.GridBuilder.html
pub trait BinsBuildingStrategy<T>
    where
        T: Ord
{
    /// Given some observations in a 1-dimensional array it returns a `BinsBuildingStrategy`
    /// that has learned the required parameter to build a collection of [`Bins`].
    ///
    /// [`Bins`]: ../struct.Bins.html
    fn from_array(array: ArrayView1<T>) -> Self;

    /// Returns a [`Bins`] instance, built accordingly to the parameters
    /// inferred from observations in [`from_array`].
    ///
    /// [`Bins`]: ../struct.Bins.html
    /// [`from_array`]: #method.from_array.html
    fn build(&self) -> Bins<T>;

    /// Returns the optimal number of bins, according to the parameters
    /// inferred from observations in [`from_array`].
    ///
    /// [`from_array`]: #method.from_array.html
    fn n_bins(&self) -> usize;
}

struct EquiSpaced<T> {
    n_bins: usize,
    min: T,
    max: T,
}

/// Square root (of data size) strategy, used by Excel and other programs
/// for its speed and simplicity.
///
/// Let `n` be the number of observations. Then
///
/// `n_bins` = `sqrt(n)`
pub struct Sqrt<T> {
    builder: EquiSpaced<T>,
}

/// A strategy that does not take variability into account, only data size. Commonly
/// overestimates number of bins required.
///
/// Let `n` be the number of observations and `n_bins` the number of bins.
///
/// `n_bins` = 2`n`<sup>1/3</sup>
///
/// `n_bins` is only proportional to cube root of `n`. It tends to overestimate
/// the `n_bins` and it does not take into account data variability.
pub struct Rice<T> {
    builder: EquiSpaced<T>,
}

/// R’s default strategy, only accounts for data size. Only optimal for gaussian data and
/// underestimates number of bins for large non-gaussian datasets.
///
/// Let `n` be the number of observations.
/// The number of bins is the base 2 log of `n`. This estimator assumes normality of data and
/// is too conservative for larger, non-normal datasets.
///
/// This is the default method in R’s hist method.
pub struct Sturges<T> {
    builder: EquiSpaced<T>,
}

/// Robust (resilient to outliers) strategy that takes into
/// account data variability and data size.
///
/// Let `n` be the number of observations.
///
/// `bin_width` = 2 * `IQR` / `n^(1/3)`
///
/// The bin width is proportional to the interquartile range ([`IQR`]) and inversely proportional to
/// cube root of `n`. It can be too conservative for small datasets, but it is quite good for
/// large datasets.
///
/// The [`IQR`] is very robust to outliers.
///
/// [`IQR`]: https://en.wikipedia.org/wiki/Interquartile_range
pub struct FreedmanDiaconis<T> {
    builder: EquiSpaced<T>,
}

enum SturgesOrFD<T> {
    Sturges(Sturges<T>),
    FreedmanDiaconis(FreedmanDiaconis<T>),
}

/// Maximum of the [`Sturges`] and [`FreedmanDiaconis`] strategies.
/// Provides good all around performance.
///
/// A compromise to get a good value. For small datasets the [`Sturges`] value will usually be chosen,
/// while larger datasets will usually default to [`FreedmanDiaconis`]. Avoids the overly
/// conservative behaviour of [`FreedmanDiaconis`] and [`Sturges`] for
/// small and large datasets respectively.
///
/// [`Sturges`]: struct.Sturges.html
/// [`FreedmanDiaconis`]: struct.FreedmanDiaconis.html
pub struct Auto<T> {
    builder: SturgesOrFD<T>,
}

impl<T> EquiSpaced<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn new(n_bins: usize, min: T, max: T) -> Self
    {
        Self { n_bins, min, max }
    }

    fn build(&self) -> Bins<T> {
        let edges = match self.n_bins {
            0 => Edges::from(vec![]),
            1 => {
                Edges::from(
                    vec![self.min.clone(), self.max.clone()]
                )
            },
            _ => {
                let bin_width = self.bin_width();
                let mut edges: Vec<T> = vec![];
                for i in 0..(self.n_bins+1) {
                    let edge = self.min.clone() + T::from_usize(i).unwrap()*bin_width.clone();
                    edges.push(edge);
                }
                Edges::from(edges)
            },
        };
        Bins::new(edges)
    }

    fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// The bin width (or bin length) according to the fitted strategy.
    fn bin_width(&self) -> T {
        let range = self.max.clone() - self.min.clone();
        let bin_width = range / T::from_usize(self.n_bins).unwrap();
        bin_width
    }
}

impl<T> BinsBuildingStrategy<T> for Sqrt<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array(a: ArrayView1<T>) -> Self
    {
        let n_elems = a.len();
        let n_bins = (n_elems as f64).sqrt().round() as usize;
        let min = a.min().clone();
        let max = a.max().clone();
        let builder = EquiSpaced::new(n_bins, min, max);
        Self { builder }
    }

    fn build(&self) -> Bins<T> {
        self.builder.build()
    }

    fn n_bins(&self) -> usize {
        self.builder.n_bins()
    }
}

impl<T> Sqrt<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    /// The bin width (or bin length) according to the fitted strategy.
    pub fn bin_width(&self) -> T {
        self.builder.bin_width()
    }
}

impl<T> BinsBuildingStrategy<T> for Rice<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array(a: ArrayView1<T>) -> Self
    {
        let n_elems = a.len();
        let n_bins = (2.*n_elems as f64).powf(1./3.).round() as usize;
        let min = a.min().clone();
        let max = a.max().clone();
        let builder = EquiSpaced::new(n_bins, min, max);
        Self { builder }
    }

    fn build(&self) -> Bins<T> {
        self.builder.build()
    }

    fn n_bins(&self) -> usize {
        self.builder.n_bins()
    }
}

impl<T> Rice<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    /// The bin width (or bin length) according to the fitted strategy.
    pub fn bin_width(&self) -> T {
        self.builder.bin_width()
    }
}

impl<T> BinsBuildingStrategy<T> for Sturges<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array(a: ArrayView1<T>) -> Self
    {
        let n_elems = a.len();
        let n_bins = (n_elems as f64).log2().round() as usize + 1;
        let min = a.min().clone();
        let max = a.max().clone();
        let builder = EquiSpaced::new(n_bins, min, max);
        Self { builder }
    }

    fn build(&self) -> Bins<T> {
        self.builder.build()
    }

    fn n_bins(&self) -> usize {
        self.builder.n_bins()
    }
}

impl<T> Sturges<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    /// The bin width (or bin length) according to the fitted strategy.
    pub fn bin_width(&self) -> T {
        self.builder.bin_width()
    }
}

impl<T> BinsBuildingStrategy<T> for FreedmanDiaconis<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array(a: ArrayView1<T>) -> Self
    {
        let n_bins = a.len();

        let mut a_copy = a.to_owned();
        let first_quartile = a_copy.quantile_mut::<Nearest>(0.25);
        let third_quartile = a_copy.quantile_mut::<Nearest>(0.75);
        let iqr = third_quartile - first_quartile;

        let bin_width = FreedmanDiaconis::compute_bin_width(n_bins, iqr);
        let min = a_copy.min().clone();
        let max = a_copy.max().clone();
        let mut max_edge = min.clone();
        while max_edge < max {
            max_edge = max_edge + bin_width.clone();
        }
        let builder = EquiSpaced::new(n_bins, min, max_edge);
        Self { builder }
    }

    fn build(&self) -> Bins<T> {
        self.builder.build()
    }

    fn n_bins(&self) -> usize {
        self.builder.n_bins()
    }
}

impl<T> FreedmanDiaconis<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn compute_bin_width(n_bins: usize, iqr: T) -> T
    {
        let denominator = (n_bins as f64).powf(1. / 3.);
        let bin_width = T::from_usize(2).unwrap() * iqr / T::from_f64(denominator).unwrap();
        bin_width
    }

    /// The bin width (or bin length) according to the fitted strategy.
    pub fn bin_width(&self) -> T {
        self.builder.bin_width()
    }
}

impl<T> BinsBuildingStrategy<T> for Auto<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    fn from_array(a: ArrayView1<T>) -> Self
    {
        let fd_builder = FreedmanDiaconis::from_array(a.view());
        let sturges_builder = Sturges::from_array(a.view());
        let builder = {
            if fd_builder.bin_width() > sturges_builder.bin_width() {
                SturgesOrFD::Sturges(sturges_builder)
            } else {
                SturgesOrFD::FreedmanDiaconis(fd_builder)
            }
        };
        Self { builder }
    }

    fn build(&self) -> Bins<T> {
        // Ugly
        match &self.builder {
            SturgesOrFD::FreedmanDiaconis(b) => b.build(),
            SturgesOrFD::Sturges(b) => b.build(),
        }
    }

    fn n_bins(&self) -> usize {
        // Ugly
        match &self.builder {
            SturgesOrFD::FreedmanDiaconis(b) => b.n_bins(),
            SturgesOrFD::Sturges(b) => b.n_bins(),
        }
    }
}

impl<T> Auto<T>
    where
        T: Ord + Clone + FromPrimitive + NumOps
{
    /// The bin width (or bin length) according to the fitted strategy.
    pub fn bin_width(&self) -> T {
        // Ugly
        match &self.builder {
            SturgesOrFD::FreedmanDiaconis(b) => b.bin_width(),
            SturgesOrFD::Sturges(b) => b.bin_width(),
        }
    }
}
