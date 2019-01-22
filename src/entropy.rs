//! Summary statistics (e.g. mean, variance, etc.).
use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{FromPrimitive, Float};

/// Extension trait for `ArrayBase` providing methods
/// to compute information theory quantities
/// (e.g. entropy, Kullback–Leibler divergence, etc.).
pub trait EntropyExt<A, S, D>
    where
        S: Data<Elem = A>,
        D: Dimension,
{
    /// Computes the [entropy] *S* of the array values, defined as
    ///
    /// ```text
    ///       n
    /// S = - ∑ xᵢ ln(xᵢ)
    ///      i=1
    /// ```
    ///
    /// If the array is empty, `None` is returned.
    ///
    /// **Panics** if any element in the array is negative.
    ///
    /// ## Remarks
    ///
    /// The entropy is a measure used in [Information Theory]
    /// to describe a probability distribution: it only make sense
    /// when the array values sum to 1, with each entry between
    /// 0 and 1 (extremes included).
    ///
    /// By definition, *xᵢ ln(xᵢ)* is set to 0 if *xᵢ* is 0.
    ///
    /// [entropy]: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    /// [Information Theory]: https://en.wikipedia.org/wiki/Information_theory
    fn entropy(&self) -> Option<A>
    where
        A: Float + FromPrimitive;
}


impl<A, S, D> EntropyExt<A, S, D> for ArrayBase<S, D>
    where
        S: Data<Elem = A>,
        D: Dimension,
{
    fn entropy(&self) -> Option<A>
        where
            A: Float + FromPrimitive
    {
        unimplemented!()
    }
}
