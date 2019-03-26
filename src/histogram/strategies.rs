//! Strategies to build [`Bins`]s and [`Grid`]s (using [`GridBuilder`]) inferring
//! optimal parameters directly from data.
//!
//! The docs for each strategy have been taken almost verbatim from [`NumPy`].
//!
//! Each strategy specifies how to compute the optimal number of [`Bins`] or
//! the optimal bin width.
//! For those strategies that prescribe the optimal number
//! of [`Bins`] we then compute the optimal bin width with
//!
//! `bin_width = (max - min)/n`
//!
//! All our bins are left-inclusive and right-exclusive: we make sure to add an extra bin
//! if it is necessary to include the maximum value of the array that has been passed as argument
//! to the `from_array` method.
//!
//! [`Bins`]: ../struct.Bins.html
//! [`Grid`]: ../struct.Grid.html
//! [`GridBuilder`]: ../struct.GridBuilder.html
//! [`NumPy`]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
use super::super::interpolate::Nearest;
use super::super::{Quantile1dExt, QuantileExt};
use super::{Bins, Edges};
use super::errors::StrategyError;
use ndarray::prelude::*;
use ndarray::Data;
use num_traits::{FromPrimitive, NumOps, Zero};

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
pub trait BinsBuildingStrategy {
    type Elem: Ord;
    /// Given some observations in a 1-dimensional array it returns a `BinsBuildingStrategy`
    /// that has learned the required parameter to build a collection of [`Bins`].
    ///
    /// It returns `None` if it is not possible to build a collection of [`Bins`] given
    /// the observed data according to the chosen strategy.
    ///
    /// [`Bins`]: ../struct.Bins.html
    fn from_array<S>(array: &ArrayBase<S, Ix1>) -> Result<Option<Self>, StrategyError>
    where
        S: Data<Elem = Self::Elem>,
        Self: std::marker::Sized;

    /// Returns a [`Bins`] instance, built accordingly to the parameters
    /// inferred from observations in [`from_array`].
    ///
    /// [`Bins`]: ../struct.Bins.html
    /// [`from_array`]: #method.from_array.html
    fn build(&self) -> Bins<Self::Elem>;

    /// Returns the optimal number of bins, according to the parameters
    /// inferred from observations in [`from_array`].
    ///
    /// [`from_array`]: #method.from_array.html
    fn n_bins(&self) -> usize;
}

#[derive(Debug)]
struct EquiSpaced<T> {
    bin_width: T,
    min: T,
    max: T,
}

/// Square root (of data size) strategy, used by Excel and other programs
/// for its speed and simplicity.
///
/// Let `n` be the number of observations. Then
///
/// `n_bins` = `sqrt(n)`
#[derive(Debug)]
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
#[derive(Debug)]
pub struct Rice<T> {
    builder: EquiSpaced<T>,
}

/// R’s default strategy, only accounts for data size. Only optimal for gaussian data and
/// underestimates number of bins for large non-gaussian datasets.
///
/// Let `n` be the number of observations.
/// The number of bins is 1 plus the base 2 log of `n`. This estimator assumes normality of data and
/// is too conservative for larger, non-normal datasets.
///
/// This is the default method in R’s hist method.
#[derive(Debug)]
pub struct Sturges<T> {
    builder: EquiSpaced<T>,
}

/// Robust (resilient to outliers) strategy that takes into
/// account data variability and data size.
///
/// Let `n` be the number of observations.
///
/// `bin_width` = 2×`IQR`×`n`<sup>−1/3</sup>
///
/// The bin width is proportional to the interquartile range ([`IQR`]) and inversely proportional to
/// cube root of `n`. It can be too conservative for small datasets, but it is quite good for
/// large datasets.
///
/// The [`IQR`] is very robust to outliers.
///
/// [`IQR`]: https://en.wikipedia.org/wiki/Interquartile_range
#[derive(Debug)]
pub struct FreedmanDiaconis<T> {
    builder: EquiSpaced<T>,
}

#[derive(Debug)]
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
#[derive(Debug)]
pub struct Auto<T> {
    builder: SturgesOrFD<T>,
}

impl<T> EquiSpaced<T>
where
    T: Ord + Clone + FromPrimitive + NumOps + Zero,
{
    /// Returns `Err(StrategyError)` if `bin_width<=0` or `min` >= `max`.
    /// Returns `Ok(Self)` otherwise.
    fn new(bin_width: T, min: T, max: T) -> Result<Self, StrategyError> {
        if (bin_width <= T::zero()) | (min >= max) {
            Err(StrategyError)
        } else {
            Ok(Self {
                bin_width,
                min,
                max,
            })
        }
    }

    fn build(&self) -> Bins<T> {
        let n_bins = self.n_bins();
        let mut edges: Vec<T> = vec![];
        for i in 0..(n_bins + 1) {
            let edge = self.min.clone() + T::from_usize(i).unwrap() * self.bin_width.clone();
            edges.push(edge);
        }
        Bins::new(Edges::from(edges))
    }

    fn n_bins(&self) -> usize {
        let mut max_edge = self.min.clone();
        let mut n_bins = 0;
        while max_edge <= self.max {
            max_edge = max_edge + self.bin_width.clone();
            n_bins += 1;
        }
        return n_bins;
    }

    fn bin_width(&self) -> T {
        self.bin_width.clone()
    }
}

impl<T> BinsBuildingStrategy for Sqrt<T>
where
    T: Ord + Clone + FromPrimitive + NumOps + Zero,
{
    type Elem = T;

    /// Returns `Err(StrategyError)` if the array is constant.
    /// Returns `Ok(None)` if `a.len()==0`.
    /// Returns `Ok(Self)` otherwise.
    fn from_array<S>(a: &ArrayBase<S, Ix1>) -> Result<Option<Self>, StrategyError>
    where
        S: Data<Elem = Self::Elem>,
    {
        let n_elems = a.len();
        let n_bins = (n_elems as f64).sqrt().round() as usize;
        match (a.min(), a.max()) {
            (Some(min), Some(max)) => {
                let bin_width = compute_bin_width(min.clone(), max.clone(), n_bins);
                let builder = EquiSpaced::new(bin_width, min.clone(), max.clone())?;
                Ok(Some(Self { builder }))
            }
            _ => Ok(None),
        }
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
    T: Ord + Clone + FromPrimitive + NumOps + Zero,
{
    /// The bin width (or bin length) according to the fitted strategy.
    pub fn bin_width(&self) -> T {
        self.builder.bin_width()
    }
}

impl<T> BinsBuildingStrategy for Rice<T>
where
    T: Ord + Clone + FromPrimitive + NumOps + Zero,
{
    type Elem = T;

    /// Returns `Err(StrategyError)` if the array is constant.
    /// Returns `Ok(None)` if `a.len()==0`.
    /// Returns `Ok(Self)` otherwise.
    fn from_array<S>(a: &ArrayBase<S, Ix1>) -> Result<Option<Self>, StrategyError>
    where
        S: Data<Elem = Self::Elem>,
    {
        let n_elems = a.len();
        let n_bins = (2. * (n_elems as f64).powf(1. / 3.)).round() as usize;
        match (a.min(), a.max()) {
            (Some(min), Some(max)) => {
                let bin_width = compute_bin_width(min.clone(), max.clone(), n_bins);
                let builder = EquiSpaced::new(bin_width, min.clone(), max.clone())?;
                Ok(Some(Self { builder }))
            }
            _ => Ok(None),
        }
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
    T: Ord + Clone + FromPrimitive + NumOps + Zero,
{
    /// The bin width (or bin length) according to the fitted strategy.
    pub fn bin_width(&self) -> T {
        self.builder.bin_width()
    }
}

impl<T> BinsBuildingStrategy for Sturges<T>
where
    T: Ord + Clone + FromPrimitive + NumOps + Zero,
{
    type Elem = T;

    /// Returns `Err(StrategyError)` if the array is constant.
    /// Returns `Ok(None)` if `a.len()==0`.
    /// Returns `Ok(Self)` otherwise.
    fn from_array<S>(a: &ArrayBase<S, Ix1>) -> Result<Option<Self>, StrategyError>
    where
        S: Data<Elem = Self::Elem>,
    {
        let n_elems = a.len();
        let n_bins = (n_elems as f64).log2().round() as usize + 1;
        match (a.min(), a.max()) {
            (Some(min), Some(max)) => {
                let bin_width = compute_bin_width(min.clone(), max.clone(), n_bins);
                let builder = EquiSpaced::new(bin_width, min.clone(), max.clone())?;
                Ok(Some(Self { builder }))
            }
            _ => Ok(None),
        }
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
    T: Ord + Clone + FromPrimitive + NumOps + Zero,
{
    /// The bin width (or bin length) according to the fitted strategy.
    pub fn bin_width(&self) -> T {
        self.builder.bin_width()
    }
}

impl<T> BinsBuildingStrategy for FreedmanDiaconis<T>
where
    T: Ord + Clone + FromPrimitive + NumOps + Zero,
{
    type Elem = T;

    /// Returns `Err(StrategyError)` if `IQR==0`.
    /// Returns `Ok(None)` if `a.len()==0`.
    /// Returns `Ok(Self)` otherwise.
    fn from_array<S>(a: &ArrayBase<S, Ix1>) -> Result<Option<Self>, StrategyError>
    where
        S: Data<Elem = Self::Elem>,
    {
        let n_points = a.len();
        if n_points == 0 {
            return Ok(None);
        }

        let mut a_copy = a.to_owned();
        let first_quartile = a_copy.quantile_mut::<Nearest>(0.25).unwrap();
        let third_quartile = a_copy.quantile_mut::<Nearest>(0.75).unwrap();
        let iqr = third_quartile - first_quartile;

        let bin_width = FreedmanDiaconis::compute_bin_width(n_points, iqr);
        match (a.min(), a.max()) {
            (Some(min), Some(max)) => {
                let builder = EquiSpaced::new(bin_width, min.clone(), max.clone())?;
                Ok(Some(Self { builder }))
            }
            _ => Ok(None),
        }
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
    T: Ord + Clone + FromPrimitive + NumOps + Zero,
{
    fn compute_bin_width(n_bins: usize, iqr: T) -> T {
        let denominator = (n_bins as f64).powf(1. / 3.);
        let bin_width = T::from_usize(2).unwrap() * iqr / T::from_f64(denominator).unwrap();
        bin_width
    }

    /// The bin width (or bin length) according to the fitted strategy.
    pub fn bin_width(&self) -> T {
        self.builder.bin_width()
    }
}

impl<T> BinsBuildingStrategy for Auto<T>
where
    T: Ord + Clone + FromPrimitive + NumOps + Zero,
{
    type Elem = T;

    /// Returns `Err(StrategyError)` if `IQR==0`.
    /// Returns `Ok(None)` if `a.len()==0`.
    /// Returns `Ok(Self)` otherwise.
    fn from_array<S>(a: &ArrayBase<S, Ix1>) -> Result<Option<Self>, StrategyError>
    where
        S: Data<Elem = Self::Elem>,
    {
        let fd_builder = FreedmanDiaconis::from_array(&a);
        let sturges_builder = Sturges::from_array(&a);
        match (fd_builder, sturges_builder) {
            (Err(_), Err(_)) => Err(StrategyError),
            (Ok(None), Ok(None)) => Ok(None),
            (Ok(None), Ok(Some(sturges_builder))) | (Err(_), Ok(Some(sturges_builder))) => {
                let builder = SturgesOrFD::Sturges(sturges_builder);
                Ok(Some(Self { builder }))
            }
            (Ok(Some(fd_builder)), Ok(None)) | (Ok(Some(fd_builder)), Err(_)) => {
                let builder = SturgesOrFD::FreedmanDiaconis(fd_builder);
                Ok(Some(Self { builder }))
            }
            (Ok(Some(fd_builder)), Ok(Some(sturges_builder))) => {
                let builder = if fd_builder.bin_width() > sturges_builder.bin_width() {
                    SturgesOrFD::Sturges(sturges_builder)
                } else {
                    SturgesOrFD::FreedmanDiaconis(fd_builder)
                };
                Ok(Some(Self { builder }))
            }
        }
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
    T: Ord + Clone + FromPrimitive + NumOps + Zero,
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

/// Given a range (max, min) and the number of bins, it returns
/// the associated bin_width:
///
/// `bin_width = (max - min)/n`
///
/// **Panics** if `n_bins == 0` and division by 0 panics for `T`.
fn compute_bin_width<T>(min: T, max: T, n_bins: usize) -> T
where
    T: Ord + Clone + FromPrimitive + NumOps + Zero,
{
    let range = max.clone() - min.clone();
    let bin_width = range / T::from_usize(n_bins).unwrap();
    bin_width
}

#[cfg(test)]
mod equispaced_tests {
    use super::*;

    #[test]
    fn bin_width_has_to_be_positive() {
        assert!(EquiSpaced::new(0, 0, 200).is_none());
    }

    #[test]
    fn min_has_to_be_strictly_smaller_than_max() {
        assert!(EquiSpaced::new(10, 0, 0).is_none());
    }
}

#[cfg(test)]
mod sqrt_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn constant_array_are_bad() {
        assert!(Sqrt::from_array(&array![1, 1, 1, 1, 1, 1, 1]).is_none());
    }

    #[test]
    fn empty_arrays_cause_panic() {
        assert!(Sqrt::<usize>::from_array(&array![]).is_none());
    }
}

#[cfg(test)]
mod rice_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn constant_array_are_bad() {
        assert!(Rice::from_array(&array![1, 1, 1, 1, 1, 1, 1]).is_none());
    }

    #[test]
    fn empty_arrays_cause_panic() {
        assert!(Rice::<usize>::from_array(&array![]).is_none());
    }
}

#[cfg(test)]
mod sturges_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn constant_array_are_bad() {
        assert!(Sturges::from_array(&array![1, 1, 1, 1, 1, 1, 1]).is_none());
    }

    #[test]
    fn empty_arrays_cause_panic() {
        assert!(Sturges::<usize>::from_array(&array![]).is_none());
    }
}

#[cfg(test)]
mod fd_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn constant_array_are_bad() {
        assert!(FreedmanDiaconis::from_array(&array![1, 1, 1, 1, 1, 1, 1]).is_none());
    }

    #[test]
    fn zero_iqr_causes_panic() {
        assert!(
            FreedmanDiaconis::from_array(&array![-20, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 20])
                .is_none()
        );
    }

    #[test]
    fn empty_arrays_cause_panic() {
        assert!(FreedmanDiaconis::<usize>::from_array(&array![]).is_none());
    }
}

#[cfg(test)]
mod auto_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn constant_array_are_bad() {
        assert!(Auto::from_array(&array![1, 1, 1, 1, 1, 1, 1]).is_none());
    }

    #[test]
    fn zero_iqr_is_handled_by_sturged() {
        assert!(Auto::from_array(&array![-20, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 20]).is_some());
    }

    #[test]
    fn empty_arrays_cause_panic() {
        assert!(Auto::<usize>::from_array(&array![]).is_none());
    }
}
