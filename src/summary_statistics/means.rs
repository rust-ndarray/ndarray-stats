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

    fn central_moment(&self, order: usize) -> Option<A>
    where
        A: Float + FromPrimitive
    {
        let mean = self.mean();
        match mean {
            None => None,
            Some(mean) => {
                match order {
                    0 => Some(A::one()),
                    1 => Some(A::zero()),
                    n => {
                        let n_elements = A::from_usize(self.len()).
                            expect("Converting number of elements to `A` must not fail");
                        let shifted_array = self.map(|x| x.clone() - mean);
                        let correction_term = -shifted_array.sum() / n_elements;
                        let coefficients: Vec<A> = (0..=n).map(
                            |k| A::from_usize(binomial_coefficient(n, k)).unwrap() *
                                shifted_array.map(|x| x.powi((n - k) as i32)).sum() / n_elements
                        ).collect();
                        // Use Horner's method here
                        let mut result = A::zero();
                        for (k, coefficient) in coefficients.iter().enumerate() {
                            result = result + *coefficient * correction_term.powi(k as i32);
                        }
                        Some(result)
                    }
                }
            }
        }
    }
}

/// Returns the binomial coefficient "n over k".
fn binomial_coefficient(n: usize, k: usize) -> usize {
    // BC(n, k) = BC(n, n-k)
    let k = if k > n - k {
        n - k
    } else {
        k
    };
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i);
        result = result / (i + 1);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::SummaryStatisticsExt;
    use std::f64;
    use approx::abs_diff_eq;
    use noisy_float::types::N64;
    use ndarray::{array, Array, Array1};
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;

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

    #[test]
    fn test_central_order_moment_with_empty_array_of_floats() {
        let a: Array1<f64> = array![];
        assert!(a.central_moment(1).is_none());
    }

    #[test]
    fn test_zeroth_central_order_moment_is_one() {
        let n = 50;
        let bound: f64 = 200.;
        let a = Array::random(
            n,
            Uniform::new(-bound.abs(), bound.abs())
        );
        assert_eq!(a.central_moment(0).unwrap(), 1.);
    }

    #[test]
    fn test_first_central_order_moment_is_zero() {
        let n = 50;
        let bound: f64 = 200.;
        let a = Array::random(
            n,
            Uniform::new(-bound.abs(), bound.abs())
        );
        assert_eq!(a.central_moment(1).unwrap(), 0.);
    }

    #[test]
    fn test_central_order_moments() {
        let a: Array1<f64> = array![
            0.07820559, 0.5026185 , 0.80935324, 0.39384033, 0.9483038,
            0.62516215, 0.90772261, 0.87329831, 0.60267392, 0.2960298,
            0.02810356, 0.31911966, 0.86705506, 0.96884832, 0.2222465,
            0.42162446, 0.99909868, 0.47619762, 0.91696979, 0.9972741,
            0.09891734, 0.76934818, 0.77566862, 0.7692585 , 0.2235759,
            0.44821286, 0.79732186, 0.04804275, 0.87863238, 0.1111003,
            0.6653943 , 0.44386445, 0.2133176 , 0.39397086, 0.4374617,
            0.95896624, 0.57850146, 0.29301706, 0.02329879, 0.2123203,
            0.62005503, 0.996492  , 0.5342986 , 0.97822099, 0.5028445,
            0.6693834 , 0.14256682, 0.52724704, 0.73482372, 0.1809703,
        ];
        // Computed using scipy.stats.moment
        let expected_moments = vec![
            1., 0., 0.09339920262960291, -0.0026849636727735186,
            0.015403769257729755, -0.001204176487006564, 0.002976822584939186,
        ];
        for (order, expected_moment) in expected_moments.iter().enumerate() {
            abs_diff_eq!(a.central_moment(order).unwrap(), expected_moment, epsilon = f64::EPSILON);
        }
    }
}
