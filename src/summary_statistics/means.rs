use ndarray::{Data, Dimension, ArrayBase};
use num_traits::{FromPrimitive, Float, Zero};
use std::ops::{Add, Div};
use super::SummaryStatisticsExt;


impl<A, S, D> SummaryStatisticsExt<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn mean(&self) -> Option<A>
    where
        A: Clone + FromPrimitive + Add<Output=A> + Div<Output=A> + Zero
    {
        let n_elements = self.len();
        if n_elements == 0 {
            None
        } else {
            let n_elements = A::from_usize(n_elements)
                .expect("Converting number of elements to `A` must not fail.");
            Some(self.sum() / n_elements)
        }
    }

    fn harmonic_mean(&self) -> Option<A>
    where
        A: Float + FromPrimitive,
    {
        self.map(|x| x.recip()).mean().map(|x| x.recip())
    }

    fn geometric_mean(&self) -> Option<A>
        where
            A: Float + FromPrimitive,
    {
        self.map(|x| x.ln()).mean().map(|x| x.exp())
    }
}

#[cfg(test)]
mod tests {
    use super::SummaryStatisticsExt;
    use std::f64;
    use approx::abs_diff_eq;
    use noisy_float::types::N64;
    use ndarray::{array, Array1};

    #[test]
    fn test_means_with_nan_values() {
        let a = array![f64::NAN, 1.];
        assert!(a.mean().unwrap().is_nan());
        assert!(a.harmonic_mean().unwrap().is_nan());
        assert!(a.geometric_mean().unwrap().is_nan());
    }

    #[test]
    fn test_means_with_empty_array_of_floats() {
        let a: Array1<f64> = array![];
        assert!(a.mean().is_none());
        assert!(a.harmonic_mean().is_none());
        assert!(a.geometric_mean().is_none());
    }

    #[test]
    fn test_means_with_empty_array_of_noisy_floats() {
        let a: Array1<N64> = array![];
        assert!(a.mean().is_none());
        assert!(a.harmonic_mean().is_none());
        assert!(a.geometric_mean().is_none());
    }

    #[test]
    fn test_means_with_array_of_floats() {
        let a: Array1<f64> = array![
            0.99889651, 0.0150731 , 0.28492482, 0.83819218, 0.48413156,
            0.80710412, 0.41762936, 0.22879429, 0.43997224, 0.23831807,
            0.02416466, 0.6269962 , 0.47420614, 0.56275487, 0.78995021,
            0.16060581, 0.64635041, 0.34876609, 0.78543249, 0.19938356,
            0.34429457, 0.88072369, 0.17638164, 0.60819363, 0.250392  ,
            0.69912532, 0.78855523, 0.79140914, 0.85084218, 0.31839879,
            0.63381769, 0.22421048, 0.70760302, 0.99216018, 0.80199153,
            0.19239188, 0.61356023, 0.31505352, 0.06120481, 0.66417377,
            0.63608897, 0.84959691, 0.43599069, 0.77867775, 0.88267754,
            0.83003623, 0.67016118, 0.67547638, 0.65220036, 0.68043427
        ];
        // Computed using NumPy
        let expected_mean = 0.5475494059146699;
        // Computed using SciPy
        let expected_harmonic_mean = 0.21790094950226022;
        // Computed using SciPy
        let expected_geometric_mean = 0.4345897639796527;

        abs_diff_eq!(a.mean().unwrap(), expected_mean, epsilon = f64::EPSILON);
        abs_diff_eq!(a.harmonic_mean().unwrap(), expected_harmonic_mean, epsilon = f64::EPSILON);
        abs_diff_eq!(a.geometric_mean().unwrap(), expected_geometric_mean, epsilon = f64::EPSILON);
    }
}
