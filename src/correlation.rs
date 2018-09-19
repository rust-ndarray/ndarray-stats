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
        A: Float,
    {
        unimplemented!();
    }
}