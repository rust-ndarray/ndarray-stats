use ndarray::prelude::*;
use ndarray::Data;
use num_traits::{Float, FromPrimitive};

/// Extension trait for `ArrayBase` providing functions
/// to compute different correlation measures.
pub trait CorrelationExt<A, S>
where
    S: Data<Elem = A>,
{
    /// Return the covariance matrix `C` for a 2-dimensional
    /// array of observations `M`.
    ///
    /// Let `(r, o)` be the shape of `M`:
    /// - `r` is the number of random variables;
    /// - `o` is the number of observations we have collected
    /// for each random variable.
    ///
    /// Every column in `M` is an experiment: a single observation for each
    /// random variable.
    /// Each row in `M` contains all the observations for a certain random variable.
    ///
    /// The parameter `ddof` specifies the "delta degrees of freedom". For
    /// example, to calculate the population covariance, use `ddof = 0`, or to
    /// calculate the sample covariance (unbiased estimate), use `ddof = 1`.
    ///
    /// The covariance of two random variables is defined as:
    ///
    /// ```text
    ///                1       n
    /// cov(X, Y) = ――――――――   ∑ (xᵢ - x̅)(yᵢ - y̅)
    ///             n - ddof  i=1
    /// ```
    ///
    /// where
    ///
    /// ```text
    ///     1   n
    /// x̅ = ―   ∑ xᵢ
    ///     n  i=1
    /// ```
    /// and similarly for ̅y.
    ///
    /// **Panics** if `ddof` is greater than or equal to the number of
    /// observations, if the number of observations is zero and division by
    /// zero panics for type `A`, or if the type cast of `n_observations` from
    /// `usize` to `A` fails.
    ///
    /// # Example
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate ndarray_stats;
    /// use ndarray::{aview2, arr2};
    /// use ndarray_stats::CorrelationExt;
    ///
    /// let a = arr2(&[[1., 3., 5.],
    ///                [2., 4., 6.]]);
    /// let covariance = a.cov(1.);
    /// assert_eq!(
    ///    covariance,
    ///    aview2(&[[4., 4.], [4., 4.]])
    /// );
    /// ```
    fn cov(&self, ddof: A) -> Array2<A>
    where
        A: Float + FromPrimitive;

    /// Return the [Pearson correlation coefficients](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
    /// for a 2-dimensional array of observations `M`.
    ///
    /// Let `(r, o)` be the shape of `M`:
    /// - `r` is the number of random variables;
    /// - `o` is the number of observations we have collected
    /// for each random variable.
    ///
    /// Every column in `M` is an experiment: a single observation for each
    /// random variable.
    /// Each row in `M` contains all the observations for a certain random variable.
    ///
    /// The Pearson correlation coefficient of two random variables is defined as:
    ///
    /// ```text
    ///              cov(X, Y)
    /// rho(X, Y) = ――――――――――――
    ///             std(X)std(Y)
    /// ```
    ///
    /// Let `R` be the matrix returned by this function. Then
    /// ```text
    /// R_ij = rho(X_i, X_j)
    /// ```
    ///
    /// **Panics** if `M` is empty, if the type cast of `n_observations`
    /// from `usize` to `A` fails or if the standard deviation of one of the random
    ///
    /// # Example
    ///
    /// variables is zero and division by zero panics for type A.
    /// ```
    /// extern crate ndarray;
    /// extern crate ndarray_stats;
    /// use ndarray::arr2;
    /// use ndarray_stats::CorrelationExt;
    ///
    /// let a = arr2(&[[1., 3., 5.],
    ///                [2., 4., 6.]]);
    /// let corr = a.pearson_correlation();
    /// assert!(
    ///     corr.all_close(
    ///         &arr2(&[
    ///             [1., 1.],
    ///             [1., 1.],
    ///         ]),
    ///         1e-7
    ///     )
    /// );
    /// ```
    fn pearson_correlation(&self) -> Array2<A>
    where
        A: Float + FromPrimitive;
}

impl<A: 'static, S> CorrelationExt<A, S> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
{
    fn cov(&self, ddof: A) -> Array2<A>
    where
        A: Float + FromPrimitive,
    {
        let observation_axis = Axis(1);
        let n_observations = A::from_usize(self.len_of(observation_axis)).unwrap();
        let dof =
            if ddof >= n_observations {
                panic!("`ddof` needs to be strictly smaller than the \
                        number of observations provided for each \
                        random variable!")
            } else {
                n_observations - ddof
            };
        let mean = self.mean_axis(observation_axis);
        let denoised = self - &mean.insert_axis(observation_axis);
        let covariance = denoised.dot(&denoised.t());
        covariance.mapv_into(|x| x / dof)
    }

    fn pearson_correlation(&self) -> Array2<A>
    where
        A: Float + FromPrimitive,
    {
        let observation_axis = Axis(1);
        // The ddof value doesn't matter, as long as we use the same one
        // for computing covariance and standard deviation
        // We choose -1 to avoid panicking when we only have one
        // observation per random variable (or no observations at all)
        let ddof = -A::one();
        let cov = self.cov(ddof);
        let std = self.std_axis(observation_axis, ddof).insert_axis(observation_axis);
        let std_matrix = std.dot(&std.t());
        // element-wise division
        cov / std_matrix
    }
}

#[cfg(test)]
mod cov_tests {
    use super::*;
    use rand;
    use rand::distributions::Uniform;
    use ndarray_rand::RandomExt;

    quickcheck! {
        fn constant_random_variables_have_zero_covariance_matrix(value: f64) -> bool {
            let n_random_variables = 3;
            let n_observations = 4;
            let a = Array::from_elem((n_random_variables, n_observations), value);
            a.cov(1.).all_close(
                &Array::zeros((n_random_variables, n_random_variables)),
                1e-8
            )
        }

        fn covariance_matrix_is_symmetric(bound: f64) -> bool {
            let n_random_variables = 3;
            let n_observations = 4;
            let a = Array::random(
                (n_random_variables, n_observations),
                Uniform::new(-bound.abs(), bound.abs())
            );
            let covariance = a.cov(1.);
            covariance.all_close(&covariance.t(), 1e-8)
        }
    }

    #[test]
    #[should_panic]
    fn test_invalid_ddof() {
        let n_random_variables = 3;
        let n_observations = 4;
        let a = Array::random(
            (n_random_variables, n_observations),
            Uniform::new(0., 10.)
        );
        let invalid_ddof = (n_observations as f64) + rand::random::<f64>().abs();
        a.cov(invalid_ddof);
    }

    #[test]
    fn test_covariance_zero_variables() {
        let a = Array2::<f32>::zeros((0, 2));
        let cov = a.cov(1.);
        assert_eq!(cov.shape(), &[0, 0]);
    }

    #[test]
    fn test_covariance_zero_observations() {
        let a = Array2::<f32>::zeros((2, 0));
        // Negative ddof (-1 < 0) to avoid invalid-ddof panic
        let cov = a.cov(-1.);
        assert_eq!(cov.shape(), &[2, 2]);
        cov.mapv(|x| assert_eq!(x, 0.));
    }

    #[test]
    fn test_covariance_zero_variables_zero_observations() {
        let a = Array2::<f32>::zeros((0, 0));
        // Negative ddof (-1 < 0) to avoid invalid-ddof panic
        let cov = a.cov(-1.);
        assert_eq!(cov.shape(), &[0, 0]);
    }

    #[test]
    fn test_covariance_for_random_array() {
        let a = array![
            [ 0.72009497,  0.12568055,  0.55705966,  0.5959984 ,  0.69471457],
            [ 0.56717131,  0.47619486,  0.21526298,  0.88915366,  0.91971245],
            [ 0.59044195,  0.10720363,  0.76573717,  0.54693675,  0.95923036],
            [ 0.24102952,  0.131347,  0.11118028,  0.21451351,  0.30515539],
            [ 0.26952473,  0.93079841,  0.8080893 ,  0.42814155,  0.24642258]
        ];
        let numpy_covariance = array![
            [ 0.05786248,  0.02614063,  0.06446215,  0.01285105, -0.06443992],
            [ 0.02614063,  0.08733569,  0.02436933,  0.01977437, -0.06715555],
            [ 0.06446215,  0.02436933,  0.10052129,  0.01393589, -0.06129912],
            [ 0.01285105,  0.01977437,  0.01393589,  0.00638795, -0.02355557],
            [-0.06443992, -0.06715555, -0.06129912, -0.02355557,  0.09909855]
        ];
        assert_eq!(a.ndim(), 2);
        assert!(
            a.cov(1.).all_close(
                &numpy_covariance,
                1e-8
            )
        );
    }

    #[test]
    #[should_panic]
    // We lose precision, hence the failing assert
    fn test_covariance_for_badly_conditioned_array() {
        let a: Array2<f64> = array![
            [ 1e12 + 1.,  1e12 - 1.],
            [ 1e-6 + 1e-12,  1e-6 - 1e-12],
        ];
        let expected_covariance = array![
            [2., 2e-12], [2e-12, 2e-24]
        ];
        assert!(
            a.cov(1.).all_close(
                &expected_covariance,
                1e-24
            )
        );
    }
}

#[cfg(test)]
mod pearson_correlation_tests {
    use super::*;
    use rand::distributions::Uniform;
    use ndarray_rand::RandomExt;

    quickcheck! {
        fn output_matrix_is_symmetric(bound: f64) -> bool {
            let n_random_variables = 3;
            let n_observations = 4;
            let a = Array::random(
                (n_random_variables, n_observations),
                Uniform::new(-bound.abs(), bound.abs())
            );
            let pearson_correlation = a.pearson_correlation();
            pearson_correlation.all_close(&pearson_correlation.t(), 1e-8)
        }

        fn constant_random_variables_have_nan_correlation(value: f64) -> bool {
            let n_random_variables = 3;
            let n_observations = 4;
            let a = Array::from_elem((n_random_variables, n_observations), value);
            let pearson_correlation = a.pearson_correlation();
            pearson_correlation.iter().map(|x| x.is_nan()).fold(true, |acc, flag| acc & flag)
        }
    }

    #[test]
    fn test_zero_variables() {
        let a = Array2::<f32>::zeros((0, 2));
        let pearson_correlation = a.pearson_correlation();
        assert_eq!(pearson_correlation.shape(), &[0, 0]);
    }

    #[test]
    fn test_zero_observations() {
        let a = Array2::<f32>::zeros((2, 0));
        let pearson = a.pearson_correlation();
        pearson.mapv(|x| x.is_nan());
    }

    #[test]
    fn test_zero_variables_zero_observations() {
        let a = Array2::<f32>::zeros((0, 0));
        let pearson = a.pearson_correlation();
        assert_eq!(pearson.shape(), &[0, 0]);
    }

    #[test]
    fn test_for_random_array() {
        let a = array![
            [0.16351516, 0.56863268, 0.16924196, 0.72579120],
            [0.44342453, 0.19834387, 0.25411802, 0.62462382],
            [0.97162731, 0.29958849, 0.17338142, 0.80198342],
            [0.91727132, 0.79817799, 0.62237124, 0.38970998],
            [0.26979716, 0.20887228, 0.95454999, 0.96290785]
        ];
        let numpy_corrcoeff = array![
            [ 1.        ,  0.38089376,  0.08122504, -0.59931623,  0.1365648 ],
            [ 0.38089376,  1.        ,  0.80918429, -0.52615195,  0.38954398],
            [ 0.08122504,  0.80918429,  1.        ,  0.07134906, -0.17324776],
            [-0.59931623, -0.52615195,  0.07134906,  1.        , -0.8743213 ],
            [ 0.1365648 ,  0.38954398, -0.17324776, -0.8743213 ,  1.        ]
        ];
        assert_eq!(a.ndim(), 2);
        assert!(
            a.pearson_correlation().all_close(
                &numpy_corrcoeff,
                1e-7
            )
        );
    }

}
