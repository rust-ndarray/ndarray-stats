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

    #[test]
    fn test_constant_case() {
        let n_random_variables = 3;
        let n_observations = 4;
        let a = Array::from_elem((n_random_variables, n_observations), 7.);
        assert!(
            a.cov(1.).all_close(
                &Array::zeros((n_random_variables, n_random_variables)),
                1e-8
            )
        );
    }
}