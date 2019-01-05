use ndarray::{Data, Dimension, ArrayBase};
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
}


impl<A, S, D> SummaryStatisticsExt<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn mean(&self) -> Result<A, &'static str>
    where
        A: Clone + FromPrimitive + Add<Output=A> + Div<Output=A> + PartialEq + Zero
    {
        let n_elements = self.len();
        if n_elements == 0 {
            Err("The mean of an empty array is not defined.")
        } else {
            let n_elements = A::from_usize(n_elements)
                .expect("Converting number of elements to `A` must not fail.");
            Ok(self.sum() / n_elements)
        }
    }

    fn harmonic_mean(&self) -> Result<A, &'static str>
    where
        A: Float + FromPrimitive + PartialEq,
    {
        if self.len() == 0 {
            Err("The harmonic mean of an empty array is not defined.")
        } else {
            Ok(self.map(|x| x.recip()).mean()?.recip())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;
    use noisy_float::types::N64;
    use ndarray::Array1;

    #[test]
    fn test_mean_with_nan_values() {
        let a = array![f64::NAN, 1.];
        assert!(a.mean().unwrap().is_nan());
    }

    #[test]
    fn test_mean_with_empty_array_of_floats() {
        let a: Array1<f64> = array![];
        assert!(a.mean().is_err());
    }

    #[test]
    fn test_mean_with_empty_array_of_noisy_floats() {
        let a: Array1<N64> = array![];
        assert!(a.mean().is_err());
    }
}
