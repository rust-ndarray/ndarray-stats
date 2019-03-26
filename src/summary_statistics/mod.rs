//! Summary statistics (e.g. mean, variance, etc.).
use ndarray::{Data, Dimension};
use num_traits::{Float, FromPrimitive, Zero};
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
    /// If the array is empty, `None` is returned.
    ///
    /// **Panics** if `A::from_usize()` fails to convert the number of elements in the array.
    ///
    /// [`arithmetic mean`]: https://en.wikipedia.org/wiki/Arithmetic_mean
    fn mean(&self) -> Option<A>
    where
        A: Clone + FromPrimitive + Add<Output = A> + Div<Output = A> + Zero;

    /// Returns the [`harmonic mean`] `HM(X)` of all elements in the array:
    ///
    /// ```text
    ///           ⎛ n     ⎞⁻¹
    /// HM(X) = n ⎜ ∑ xᵢ⁻¹⎟
    ///           ⎝i=1    ⎠
    /// ```
    ///
    /// If the array is empty, `None` is returned.
    ///
    /// **Panics** if `A::from_usize()` fails to convert the number of elements in the array.
    ///
    /// [`harmonic mean`]: https://en.wikipedia.org/wiki/Harmonic_mean
    fn harmonic_mean(&self) -> Option<A>
    where
        A: Float + FromPrimitive;

    /// Returns the [`geometric mean`] `GM(X)` of all elements in the array:
    ///
    /// ```text
    ///         ⎛ n   ⎞¹⁄ₙ
    /// GM(X) = ⎜ ∏ xᵢ⎟
    ///         ⎝i=1  ⎠
    /// ```
    ///
    /// If the array is empty, `None` is returned.
    ///
    /// **Panics** if `A::from_usize()` fails to convert the number of elements in the array.
    ///
    /// [`geometric mean`]: https://en.wikipedia.org/wiki/Geometric_mean
    fn geometric_mean(&self) -> Option<A>
    where
        A: Float + FromPrimitive;
}

mod means;
