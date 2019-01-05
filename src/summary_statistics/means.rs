use ndarray::{Data, Dimension, ArrayBase};
use std::ops::{Add, Div};
use num_traits::{FromPrimitive, Float, Zero};


/// Extension trait for `ArrayBase` providing methods
/// to compute several summary statistics (e.g. mean, variance, etc.).
pub trait SummaryStatisticsExt<A, S, D>
    where
        S: Data<Elem = A>,
        D: Dimension,
{
    fn mean(&self) -> A
    where
        A: Clone + FromPrimitive + Add<Output=A> + Div<Output=A> + Zero;

    fn harmonic_mean(&self) -> A
    where
        A: Float + FromPrimitive;
}


impl<A, S, D> SummaryStatisticsExt<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn mean(&self) -> A
    where
        A: Clone + FromPrimitive + Add<Output=A> + Div<Output=A> + Zero
    {
        let n_elements = A::from_usize(self.len())
            .expect("Converting number of elements to `A` must not fail.");
        self.sum() / n_elements
    }

    fn harmonic_mean(&self) -> A
    where
        A: Float + FromPrimitive,
    {
        self.map(|x| x.recip()).mean().recip()
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
        assert!(a.mean().is_nan());
    }

    #[test]
    fn test_mean_with_empty_array_of_floats() {
        let a: Array1<f64> = array![];
        assert!(a.mean().is_nan());
    }

    #[test]
    #[should_panic] // This looks highly undesirable
    fn test_mean_with_empty_array_of_noisy_floats() {
        let a: Array1<N64> = array![];
        assert!(a.mean().is_nan());
    }
}
