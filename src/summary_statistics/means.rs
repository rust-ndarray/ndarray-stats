use super::SummaryStatisticsExt;
use crate::errors::EmptyInput;
use ndarray::{ArrayBase, Data, Dimension};
use num_integer::IterBinomial;
use num_traits::{Float, FromPrimitive, Zero};
use std::ops::{Add, Div};

impl<A, S, D> SummaryStatisticsExt<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn mean(&self) -> Result<A, EmptyInput>
    where
        A: Clone + FromPrimitive + Add<Output = A> + Div<Output = A> + Zero,
    {
        let n_elements = self.len();
        if n_elements == 0 {
            Err(EmptyInput)
        } else {
            let n_elements = A::from_usize(n_elements)
                .expect("Converting number of elements to `A` must not fail.");
            Ok(self.sum() / n_elements)
        }
    }

    fn harmonic_mean(&self) -> Result<A, EmptyInput>
    where
        A: Float + FromPrimitive,
    {
        self.map(|x| x.recip()).mean().map(|x| x.recip())
    }

    fn geometric_mean(&self) -> Result<A, EmptyInput>
    where
        A: Float + FromPrimitive,
    {
        self.map(|x| x.ln()).mean().map(|x| x.exp())
    }

    fn kurtosis(&self) -> Result<A, EmptyInput>
    where
        A: Float + FromPrimitive,
    {
        let central_moments = self.central_moments(4)?;
        Ok(central_moments[4] / central_moments[2].powi(2))
    }

    fn skewness(&self) -> Result<A, EmptyInput>
    where
        A: Float + FromPrimitive,
    {
        let central_moments = self.central_moments(3)?;
        Ok(central_moments[3] / central_moments[2].sqrt().powi(3))
    }

    fn standard_deviation(&self) -> Result<A, EmptyInput>
    where
        A: Float + FromPrimitive,
    {
        Ok(self.central_moment(2)?.sqrt())
    }

    fn central_moment(&self, order: u16) -> Result<A, EmptyInput>
    where
        A: Float + FromPrimitive,
    {
        if self.is_empty() {
            return Err(EmptyInput);
        }
        match order {
            0 => Ok(A::one()),
            1 => Ok(A::zero()),
            n => {
                let mean = self.mean().unwrap();
                let shifted_array = self.mapv(|x| x - mean);
                let shifted_moments = moments(shifted_array, n);
                let correction_term = -shifted_moments[1];

                let coefficients = central_moment_coefficients(&shifted_moments);
                Ok(horner_method(coefficients, correction_term))
            }
        }
    }

    fn central_moments(&self, order: u16) -> Result<Vec<A>, EmptyInput>
    where
        A: Float + FromPrimitive,
    {
        if self.is_empty() {
            return Err(EmptyInput);
        }
        match order {
            0 => Ok(vec![A::one()]),
            1 => Ok(vec![A::one(), A::zero()]),
            n => {
                // We only perform these operations once, and then reuse their
                // result to compute all the required moments
                let mean = self.mean().unwrap();
                let shifted_array = self.mapv(|x| x - mean);
                let shifted_moments = moments(shifted_array, n);
                let correction_term = -shifted_moments[1];

                let mut central_moments = vec![A::one(), A::zero()];
                for k in 2..=n {
                    let coefficients =
                        central_moment_coefficients(&shifted_moments[..=(k as usize)]);
                    let central_moment = horner_method(coefficients, correction_term);
                    central_moments.push(central_moment)
                }
                Ok(central_moments)
            }
        }
    }

    private_impl! {}
}

/// Returns a vector containing all moments of the array elements up to
/// *order*, where the *p*-th moment is defined as:
///
/// ```text
/// 1  n
/// ―  ∑ xᵢᵖ
/// n i=1
/// ```
///
/// The returned moments are ordered by power magnitude: 0th moment, 1st moment, etc.
///
/// **Panics** if `A::from_usize()` fails to convert the number of elements in the array.
fn moments<A, S, D>(a: ArrayBase<S, D>, order: u16) -> Vec<A>
where
    A: Float + FromPrimitive,
    S: Data<Elem = A>,
    D: Dimension,
{
    let n_elements =
        A::from_usize(a.len()).expect("Converting number of elements to `A` must not fail");
    let order = order as i32;

    // When k=0, we are raising each element to the 0th power
    // No need to waste CPU cycles going through the array
    let mut moments = vec![A::one()];

    if order >= 1 {
        // When k=1, we don't need to raise elements to the 1th power (identity)
        moments.push(a.sum() / n_elements)
    }

    for k in 2..=order {
        moments.push(a.map(|x| x.powi(k)).sum() / n_elements)
    }
    moments
}

/// Returns the coefficients in the polynomial expression to compute the *p*th
/// central moment as a function of the sample mean.
///
/// It takes as input all moments up to order *p*, ordered by power magnitude - *p* is
/// inferred to be the length of the *moments* array.
fn central_moment_coefficients<A>(moments: &[A]) -> Vec<A>
where
    A: Float + FromPrimitive,
{
    let order = moments.len();
    IterBinomial::new(order)
        .zip(moments.iter().rev())
        .map(|(binom, &moment)| A::from_usize(binom).unwrap() * moment)
        .collect()
}

/// Uses [Horner's method] to evaluate a polynomial with a single indeterminate.
///
/// Coefficients are expected to be sorted by ascending order
/// with respect to the indeterminate's exponent.
///
/// If the array is empty, `A::zero()` is returned.
///
/// Horner's method can evaluate a polynomial of order *n* with a single indeterminate
/// using only *n-1* multiplications and *n-1* sums - in terms of number of operations,
/// this is an optimal algorithm for polynomial evaluation.
///
/// [Horner's method]: https://en.wikipedia.org/wiki/Horner%27s_method
fn horner_method<A>(coefficients: Vec<A>, indeterminate: A) -> A
where
    A: Float,
{
    let mut result = A::zero();
    for coefficient in coefficients.into_iter().rev() {
        result = coefficient + indeterminate * result
    }
    result
}

#[cfg(test)]
mod tests {
    use super::SummaryStatisticsExt;
    use crate::errors::EmptyInput;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array, Array1};
    use ndarray_rand::RandomExt;
    use noisy_float::types::N64;
    use rand::distributions::Uniform;
    use std::f64;

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
        assert_eq!(a.mean(), Err(EmptyInput));
        assert_eq!(a.harmonic_mean(), Err(EmptyInput));
        assert_eq!(a.geometric_mean(), Err(EmptyInput));
    }

    #[test]
    fn test_means_with_empty_array_of_noisy_floats() {
        let a: Array1<N64> = array![];
        assert_eq!(a.mean(), Err(EmptyInput));
        assert_eq!(a.harmonic_mean(), Err(EmptyInput));
        assert_eq!(a.geometric_mean(), Err(EmptyInput));
    }

    #[test]
    fn test_means_with_array_of_floats() {
        let a: Array1<f64> = array![
            0.99889651, 0.0150731, 0.28492482, 0.83819218, 0.48413156, 0.80710412, 0.41762936,
            0.22879429, 0.43997224, 0.23831807, 0.02416466, 0.6269962, 0.47420614, 0.56275487,
            0.78995021, 0.16060581, 0.64635041, 0.34876609, 0.78543249, 0.19938356, 0.34429457,
            0.88072369, 0.17638164, 0.60819363, 0.250392, 0.69912532, 0.78855523, 0.79140914,
            0.85084218, 0.31839879, 0.63381769, 0.22421048, 0.70760302, 0.99216018, 0.80199153,
            0.19239188, 0.61356023, 0.31505352, 0.06120481, 0.66417377, 0.63608897, 0.84959691,
            0.43599069, 0.77867775, 0.88267754, 0.83003623, 0.67016118, 0.67547638, 0.65220036,
            0.68043427
        ];
        // Computed using NumPy
        let expected_mean = 0.5475494059146699;
        // Computed using SciPy
        let expected_harmonic_mean = 0.21790094950226022;
        // Computed using SciPy
        let expected_geometric_mean = 0.4345897639796527;

        assert_abs_diff_eq!(a.mean().unwrap(), expected_mean, epsilon = 1e-9);
        assert_abs_diff_eq!(
            a.harmonic_mean().unwrap(),
            expected_harmonic_mean,
            epsilon = 1e-7
        );
        assert_abs_diff_eq!(
            a.geometric_mean().unwrap(),
            expected_geometric_mean,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_central_moment_with_empty_array_of_floats() {
        let a: Array1<f64> = array![];
        for order in 0..=3 {
            assert_eq!(a.central_moment(order), Err(EmptyInput));
            assert_eq!(a.central_moments(order), Err(EmptyInput));
        }
    }

    #[test]
    fn test_zeroth_central_moment_is_one() {
        let n = 50;
        let bound: f64 = 200.;
        let a = Array::random(n, Uniform::new(-bound.abs(), bound.abs()));
        assert_eq!(a.central_moment(0).unwrap(), 1.);
    }

    #[test]
    fn test_first_central_moment_is_zero() {
        let n = 50;
        let bound: f64 = 200.;
        let a = Array::random(n, Uniform::new(-bound.abs(), bound.abs()));
        assert_eq!(a.central_moment(1).unwrap(), 0.);
    }

    #[test]
    fn test_standard_deviation() {
        let a: Array1<f64> = array![4.0, 9.0, 11.0, 12.0, 17.0, 5.0, 8.0, 12.0, 14.0];
        // Computed using numpy.std
        let expected_std = 3.9377878103709665;
        assert_abs_diff_eq!(
            a.standard_deviation().unwrap(),
            expected_std,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_central_moments() {
        let a: Array1<f64> = array![
            0.07820559, 0.5026185, 0.80935324, 0.39384033, 0.9483038, 0.62516215, 0.90772261,
            0.87329831, 0.60267392, 0.2960298, 0.02810356, 0.31911966, 0.86705506, 0.96884832,
            0.2222465, 0.42162446, 0.99909868, 0.47619762, 0.91696979, 0.9972741, 0.09891734,
            0.76934818, 0.77566862, 0.7692585, 0.2235759, 0.44821286, 0.79732186, 0.04804275,
            0.87863238, 0.1111003, 0.6653943, 0.44386445, 0.2133176, 0.39397086, 0.4374617,
            0.95896624, 0.57850146, 0.29301706, 0.02329879, 0.2123203, 0.62005503, 0.996492,
            0.5342986, 0.97822099, 0.5028445, 0.6693834, 0.14256682, 0.52724704, 0.73482372,
            0.1809703,
        ];
        // Computed using scipy.stats.moment
        let expected_moments = vec![
            1.,
            0.,
            0.09339920262960291,
            -0.0026849636727735186,
            0.015403769257729755,
            -0.001204176487006564,
            0.002976822584939186,
        ];
        for (order, expected_moment) in expected_moments.iter().enumerate() {
            assert_abs_diff_eq!(
                a.central_moment(order as u16).unwrap(),
                expected_moment,
                epsilon = 1e-8
            );
        }
    }

    #[test]
    fn test_bulk_central_moments() {
        // Test that the bulk method is coherent with the non-bulk method
        let n = 50;
        let bound: f64 = 200.;
        let a = Array::random(n, Uniform::new(-bound.abs(), bound.abs()));
        let order = 10;
        let central_moments = a.central_moments(order).unwrap();
        for i in 0..=order {
            assert_eq!(a.central_moment(i).unwrap(), central_moments[i as usize]);
        }
    }

    #[test]
    fn test_kurtosis_and_skewness_is_none_with_empty_array_of_floats() {
        let a: Array1<f64> = array![];
        assert_eq!(a.skewness(), Err(EmptyInput));
        assert_eq!(a.kurtosis(), Err(EmptyInput));
    }

    #[test]
    fn test_kurtosis_and_skewness() {
        let a: Array1<f64> = array![
            0.33310096, 0.98757449, 0.9789796, 0.96738114, 0.43545674, 0.06746873, 0.23706562,
            0.04241815, 0.38961714, 0.52421271, 0.93430327, 0.33911604, 0.05112372, 0.5013455,
            0.05291507, 0.62511183, 0.20749633, 0.22132433, 0.14734804, 0.51960608, 0.00449208,
            0.4093339, 0.2237519, 0.28070469, 0.7887231, 0.92224523, 0.43454188, 0.18335111,
            0.08646856, 0.87979847, 0.25483457, 0.99975627, 0.52712442, 0.41163279, 0.85162594,
            0.52618733, 0.75815023, 0.30640695, 0.14205781, 0.59695813, 0.851331, 0.39524328,
            0.73965373, 0.4007615, 0.02133069, 0.92899207, 0.79878191, 0.38947334, 0.22042183,
            0.77768353,
        ];
        // Computed using scipy.stats.kurtosis(a, fisher=False)
        let expected_kurtosis = 1.821933711687523;
        // Computed using scipy.stats.skew
        let expected_skewness = 0.2604785422878771;

        let kurtosis = a.kurtosis().unwrap();
        let skewness = a.skewness().unwrap();

        assert_abs_diff_eq!(kurtosis, expected_kurtosis, epsilon = 1e-12);
        assert_abs_diff_eq!(skewness, expected_skewness, epsilon = 1e-8);
    }
}
