use ndarray::prelude::*;
use ndarray::Data;
use num_traits::Float;

pub trait CorrelationExt<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn cov(&self, ddof: A) -> Array<A, D> 
    where
        A: Float;
}

impl<A, S, D> CorrelationExt<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn cov(&self, ddof: A) -> Array<A, D>
    where
        A: Float,
    {
        if self.ndim() < 2 {
            panic!(
                "We cannot compute the covariance of \
                an array with less than 2 dimensions!"
            );
        } else {
            unimplemented!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_case() {
        let a = Array::from_elem((3, 4), 7.);
        assert_eq!(a.cov(1.), Array::zeros((3, 4)));
    }

    #[test]
    #[should_panic]
    fn test_panic_for_1d_arrays() {
        let a = array!([1., 2., 3.]);
        a.cov(1.);
    }
}