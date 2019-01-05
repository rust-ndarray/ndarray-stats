//! Summary statistics (e.g. mean, variance, etc.).
use ndarray::{Data, Dimension};
use num_traits::{FromPrimitive, Float, Zero};
use std::result::Result;
use std::ops::{Add, Div};

/// Extension trait for `ArrayBase` providing methods
/// to compute several summary statistics (e.g. mean, variance, etc.).
pub trait SummaryStatisticsExt<A, S, D>
    where
        S: Data<Elem = A>,
        D: Dimension,
{
    /// Returns the [`arithmetic mean`] x̅ of all elements in the array:
    ///
    /// ```text
    ///     1   n
    /// x̅ = ―   ∑ xᵢ
    ///     n  i=1
    /// ```
    ///
    /// If the array is empty, an `Err` is returned.
    ///
    /// [`arithmetic mean`]: https://en.wikipedia.org/wiki/Arithmetic_mean
    fn mean(&self) -> Result<A, &'static str>
        where
            A: Clone + FromPrimitive + Add<Output=A> + Div<Output=A> + PartialEq + Zero;

    /// Returns the [`harmonic mean`] `HM(X)` of all elements in the array:
    ///
    /// ```text
    ///             n
    /// HM(X) = n ( ∑ xᵢ⁻¹)⁻¹
    ///            i=1
    /// ```
    ///
    /// If the array is empty, an `Err` is returned.
    ///
    /// [`harmonic mean`]: https://en.wikipedia.org/wiki/Harmonic_mean
    fn harmonic_mean(&self) -> Result<A, &'static str>
        where
            A: Float + FromPrimitive + PartialEq;

    /// Returns the [`geometric mean`] `GM(X)` of all elements in the array:
    ///
    /// ```text
    ///           n
    /// GM(X) = ( Π xᵢ)^(1/n)
    ///          i=1
    /// ```
    ///
    /// If the array is empty, an `Err` is returned.
    ///
    /// [`geometric mean`]: https://en.wikipedia.org/wiki/Geometric_mean
    fn geometric_mean(&self) -> Result<A, &'static str>
        where
            A: Float + FromPrimitive + PartialEq;
}

mod means;

