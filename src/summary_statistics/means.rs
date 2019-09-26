use super::SummaryStatisticsExt;
use crate::errors::{EmptyInput, MultiInputError, ShapeMismatch};
use ndarray::{Array, ArrayBase, Axis, Data, Dimension, Ix1, RemoveAxis};
use num_integer::IterBinomial;
use num_traits::{Float, FromPrimitive, Zero};
use std::ops::{Add, Div, Mul};

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

    fn weighted_mean(&self, weights: &Self) -> Result<A, MultiInputError>
    where
        A: Copy + Div<Output = A> + Mul<Output = A> + Zero,
    {
        return_err_if_empty!(self);
        let weighted_sum = self.weighted_sum(weights)?;
        Ok(weighted_sum / weights.sum())
    }

    fn weighted_sum(&self, weights: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: Copy + Mul<Output = A> + Zero,
    {
        return_err_unless_same_shape!(self, weights);
        Ok(self
            .iter()
            .zip(weights)
            .fold(A::zero(), |acc, (&d, &w)| acc + d * w))
    }

    fn weighted_mean_axis(
        &self,
        axis: Axis,
        weights: &ArrayBase<S, Ix1>,
    ) -> Result<Array<A, D::Smaller>, MultiInputError>
    where
        A: Copy + Div<Output = A> + Mul<Output = A> + Zero,
        D: RemoveAxis,
    {
        return_err_if_empty!(self);
        let mut weighted_sum = self.weighted_sum_axis(axis, weights)?;
        let weights_sum = weights.sum();
        weighted_sum.mapv_inplace(|v| v / weights_sum);
        Ok(weighted_sum)
    }

    fn weighted_sum_axis(
        &self,
        axis: Axis,
        weights: &ArrayBase<S, Ix1>,
    ) -> Result<Array<A, D::Smaller>, MultiInputError>
    where
        A: Copy + Mul<Output = A> + Zero,
        D: RemoveAxis,
    {
        if self.shape()[axis.index()] != weights.len() {
            return Err(MultiInputError::ShapeMismatch(ShapeMismatch {
                first_shape: self.shape().to_vec(),
                second_shape: weights.shape().to_vec(),
            }));
        }

        // We could use `lane.weighted_sum` here, but we're avoiding 2
        // conditions and an unwrap per lane.
        Ok(self.map_axis(axis, |lane| {
            lane.iter()
                .zip(weights)
                .fold(A::zero(), |acc, (&d, &w)| acc + d * w)
        }))
    }

    fn harmonic_mean(&self) -> Result<A, EmptyInput>
    where
        A: Float + FromPrimitive,
    {
        self.map(|x| x.recip())
            .mean()
            .map(|x| x.recip())
            .ok_or(EmptyInput)
    }

    fn geometric_mean(&self) -> Result<A, EmptyInput>
    where
        A: Float + FromPrimitive,
    {
        self.map(|x| x.ln())
            .mean()
            .map(|x| x.exp())
            .ok_or(EmptyInput)
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
