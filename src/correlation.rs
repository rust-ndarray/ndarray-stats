use ndarray::prelude::*;
use ndarray::Data;
use num_traits::{Float, FromPrimitive};

pub trait CorrelationExt<A, S>
where
    S: Data<Elem = A>,
{
    fn cov(&self, ddof: A) -> Array2<A> 
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;
    use rand::distributions::Range;
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
                Range::new(-bound.abs(), bound.abs())
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
            Range::new(0., 10.)
        );
        let invalid_ddof = (n_observations as f64) + rand::random::<f64>().abs();
        a.cov(invalid_ddof);
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
}