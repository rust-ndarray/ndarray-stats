//! Summary statistics (e.g. mean, variance, etc.).
use crate::errors::{EmptyInput, MultiInputError};
use ndarray::{Array, ArrayBase, Axis, Data, Dimension, Ix1, RemoveAxis};
use num_traits::{Float, FromPrimitive, Zero};
use std::ops::{Add, AddAssign, Div, Mul};

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
    /// If the array is empty, `Err(EmptyInput)` is returned.
    ///
    /// **Panics** if `A::from_usize()` fails to convert the number of elements in the array.
    ///
    /// [`arithmetic mean`]: https://en.wikipedia.org/wiki/Arithmetic_mean
    fn mean(&self) -> Result<A, EmptyInput>
    where
        A: Clone + FromPrimitive + Add<Output = A> + Div<Output = A> + Zero;

    /// Returns the [`arithmetic weighted mean`] x̅ of all elements in the array. Use `weighted_sum`
    /// if the `weights` are normalized (they sum up to 1.0).
    ///
    /// ```text
    ///       n
    ///       ∑ wᵢxᵢ
    ///      i=1
    /// x̅ = ―――――――――
    ///        n
    ///        ∑ wᵢ
    ///       i=1
    /// ```
    ///
    /// **Panics** if division by zero panics for type A.
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::EmptyInput` if `self` is empty
    /// * `MultiInputError::ShapeMismatch` if `self` and `weights` don't have the same shape
    ///
    /// [`arithmetic weighted mean`] https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    fn weighted_mean(&self, weights: &Self) -> Result<A, MultiInputError>
    where
        A: Copy + Div<Output = A> + Mul<Output = A> + Zero;

    /// Returns the weighted sum of all elements in the array, that is, the dot product of the
    /// arrays `self` and `weights`. Equivalent to `weighted_mean` if the `weights` are normalized.
    ///
    /// ```text
    ///      n
    /// x̅ =  ∑ wᵢxᵢ
    ///     i=1
    /// ```
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::ShapeMismatch` if `self` and `weights` don't have the same shape
    fn weighted_sum(&self, weights: &Self) -> Result<A, MultiInputError>
    where
        A: Copy + Mul<Output = A> + Zero;

    /// Returns the [`arithmetic weighted mean`] x̅ along `axis`. Use `weighted_mean_axis ` if the
    /// `weights` are normalized.
    ///
    /// ```text
    ///       n
    ///       ∑ wᵢxᵢ
    ///      i=1
    /// x̅ = ―――――――――
    ///        n
    ///        ∑ wᵢ
    ///       i=1
    /// ```
    ///
    /// **Panics** if `axis` is out of bounds.
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::EmptyInput` if `self` is empty
    /// * `MultiInputError::ShapeMismatch` if `self` length along axis is not equal to `weights` length
    ///
    /// [`arithmetic weighted mean`] https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    fn weighted_mean_axis(
        &self,
        axis: Axis,
        weights: &ArrayBase<S, Ix1>,
    ) -> Result<Array<A, D::Smaller>, MultiInputError>
    where
        A: Copy + Div<Output = A> + Mul<Output = A> + Zero,
        D: RemoveAxis;

    /// Returns the weighted sum along `axis`, that is, the dot product of `weights` and each lane
    /// of `self` along `axis`. Equivalent to `weighted_mean_axis` if the `weights` are normalized.
    ///
    /// ```text
    ///      n
    /// x̅ =  ∑ wᵢxᵢ
    ///     i=1
    /// ```
    ///
    /// **Panics** if `axis` is out of bounds.
    ///
    /// The following **errors** may be returned
    ///
    /// * `MultiInputError::ShapeMismatch` if `self` and `weights` don't have the same shape
    fn weighted_sum_axis(
        &self,
        axis: Axis,
        weights: &ArrayBase<S, Ix1>,
    ) -> Result<Array<A, D::Smaller>, MultiInputError>
    where
        A: Copy + Mul<Output = A> + Zero,
        D: RemoveAxis;

    /// Returns the [`harmonic mean`] `HM(X)` of all elements in the array:
    ///
    /// ```text
    ///           ⎛ n     ⎞⁻¹
    /// HM(X) = n ⎜ ∑ xᵢ⁻¹⎟
    ///           ⎝i=1    ⎠
    /// ```
    ///
    /// If the array is empty, `Err(EmptyInput)` is returned.
    ///
    /// **Panics** if `A::from_usize()` fails to convert the number of elements in the array.
    ///
    /// [`harmonic mean`]: https://en.wikipedia.org/wiki/Harmonic_mean
    fn harmonic_mean(&self) -> Result<A, EmptyInput>
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
    /// If the array is empty, `Err(EmptyInput)` is returned.
    ///
    /// **Panics** if `A::from_usize()` fails to convert the number of elements in the array.
    ///
    /// [`geometric mean`]: https://en.wikipedia.org/wiki/Geometric_mean
    fn geometric_mean(&self) -> Result<A, EmptyInput>
    where
        A: Float + FromPrimitive;

    /// Return weighted variance of all elements in the array.
    ///
    /// The weighted variance is computed using the [`West, D. H. D.`] incremental algorithm.
    /// Equivalent to `var_axis` if the `weights` are normalized.
    ///
    /// The parameter `ddof` specifies the "delta degrees of freedom". For example, to calculate the
    /// population variance, use `ddof = 0`, or to calculate the sample variance, use `ddof = 1`.
    ///
    /// **Panics** if `ddof` is less than zero or greater than one, or if `axis` is out of bounds,
    /// or if `A::from_usize()` fails for zero or one.
    ///
    /// [`West, D. H. D.`]: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm
    fn weighted_var(&self, weights: &Self, ddof: A) -> Result<A, MultiInputError>
    where
        A: AddAssign + Float + FromPrimitive;

    /// Return weighted standard deviation of all elements in the array.
    ///
    /// The weighted standard deviation is computed using the [`West, D. H. D.`] incremental
    /// algorithm. Equivalent to `var_axis` if the `weights` are normalized.
    ///
    /// The parameter `ddof` specifies the "delta degrees of freedom". For example, to calculate the
    /// population variance, use `ddof = 0`, or to calculate the sample variance, use `ddof = 1`.
    ///
    /// **Panics** if `ddof` is less than zero or greater than one, or if `axis` is out of bounds,
    /// or if `A::from_usize()` fails for zero or one.
    ///
    /// [`West, D. H. D.`]: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm
    fn weighted_std(&self, weights: &Self, ddof: A) -> Result<A, MultiInputError>
    where
        A: AddAssign + Float + FromPrimitive;

    /// Return weighted variance along `axis`.
    ///
    /// The weighted variance is computed using the [`West, D. H. D.`] incremental algorithm.
    /// Equivalent to `var_axis` if the `weights` are normalized.
    ///
    /// The parameter `ddof` specifies the "delta degrees of freedom". For example, to calculate the
    /// population variance, use `ddof = 0`, or to calculate the sample variance, use `ddof = 1`.
    ///
    /// **Panics** if `ddof` is less than zero or greater than one, or if `axis` is out of bounds,
    /// or if `A::from_usize()` fails for zero or one.
    ///
    /// [`West, D. H. D.`]: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm
    fn weighted_var_axis(
        &self,
        axis: Axis,
        weights: &ArrayBase<S, Ix1>,
        ddof: A,
    ) -> Result<Array<A, D::Smaller>, MultiInputError>
    where
        A: AddAssign + Float + FromPrimitive,
        D: RemoveAxis;

    /// Return weighted standard deviation along `axis`.
    ///
    /// The weighted standard deviation is computed using the [`West, D. H. D.`] incremental
    /// algorithm. Equivalent to `var_axis` if the `weights` are normalized.
    ///
    /// The parameter `ddof` specifies the "delta degrees of freedom". For example, to calculate the
    /// population variance, use `ddof = 0`, or to calculate the sample variance, use `ddof = 1`.
    ///
    /// **Panics** if `ddof` is less than zero or greater than one, or if `axis` is out of bounds,
    /// or if `A::from_usize()` fails for zero or one.
    ///
    /// [`West, D. H. D.`]: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm
    fn weighted_std_axis(
        &self,
        axis: Axis,
        weights: &ArrayBase<S, Ix1>,
        ddof: A,
    ) -> Result<Array<A, D::Smaller>, MultiInputError>
    where
        A: AddAssign + Float + FromPrimitive,
        D: RemoveAxis;

    /// Returns the [kurtosis] `Kurt[X]` of all elements in the array:
    ///
    /// ```text
    /// Kurt[X] = μ₄ / σ⁴
    /// ```
    ///
    /// where μ₄ is the fourth central moment and σ is the standard deviation of
    /// the elements in the array.
    ///
    /// This is sometimes referred to as _Pearson's kurtosis_. Fisher's kurtosis can be
    /// computed by subtracting 3 from Pearson's kurtosis.
    ///
    /// If the array is empty, `Err(EmptyInput)` is returned.
    ///
    /// **Panics** if `A::from_usize()` fails to convert the number of elements in the array.
    ///
    /// [kurtosis]: https://en.wikipedia.org/wiki/Kurtosis
    fn kurtosis(&self) -> Result<A, EmptyInput>
    where
        A: Float + FromPrimitive;

    /// Returns the [Pearson's moment coefficient of skewness] γ₁ of all elements in the array:
    ///
    /// ```text
    /// γ₁ = μ₃ / σ³
    /// ```
    ///
    /// where μ₃ is the third central moment and σ is the standard deviation of
    /// the elements in the array.
    ///
    /// If the array is empty, `Err(EmptyInput)` is returned.
    ///
    /// **Panics** if `A::from_usize()` fails to convert the number of elements in the array.
    ///
    /// [Pearson's moment coefficient of skewness]: https://en.wikipedia.org/wiki/Skewness
    fn skewness(&self) -> Result<A, EmptyInput>
    where
        A: Float + FromPrimitive;

    /// Returns the *p*-th [central moment] of all elements in the array, μₚ:
    ///
    /// ```text
    ///      1  n
    /// μₚ = ―  ∑ (xᵢ-x̅)ᵖ
    ///      n i=1
    /// ```
    ///
    /// If the array is empty, `Err(EmptyInput)` is returned.
    ///
    /// The *p*-th central moment is computed using a corrected two-pass algorithm (see Section 3.5
    /// in [Pébay et al., 2016]). Complexity is *O(np)* when *n >> p*, *p > 1*.
    ///
    /// **Panics** if `A::from_usize()` fails to convert the number of elements
    /// in the array or if `order` overflows `i32`.
    ///
    /// [central moment]: https://en.wikipedia.org/wiki/Central_moment
    /// [Pébay et al., 2016]: https://www.osti.gov/pages/servlets/purl/1427275
    fn central_moment(&self, order: u16) -> Result<A, EmptyInput>
    where
        A: Float + FromPrimitive;

    /// Returns the first *p* [central moments] of all elements in the array, see [central moment]
    /// for more details.
    ///
    /// If the array is empty, `Err(EmptyInput)` is returned.
    ///
    /// This method reuses the intermediate steps for the *k*-th moment to compute the *(k+1)*-th,
    /// being thus more efficient than repeated calls to [central moment] if the computation
    /// of central moments of multiple orders is required.
    ///
    /// **Panics** if `A::from_usize()` fails to convert the number of elements
    /// in the array or if `order` overflows `i32`.
    ///
    /// [central moments]: https://en.wikipedia.org/wiki/Central_moment
    /// [central moment]: #tymethod.central_moment
    fn central_moments(&self, order: u16) -> Result<Vec<A>, EmptyInput>
    where
        A: Float + FromPrimitive;

    private_decl! {}
}

mod means;
