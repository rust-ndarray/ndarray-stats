use ndarray::{ArrayBase, Data, Dimension, Zip};
use num_traits::{Signed, ToPrimitive};
use std::convert::Into;
use std::ops::AddAssign;

use crate::errors::{MultiInputError, ShapeMismatch};

/// An extension trait for `ArrayBase` providing functions
/// to compute different deviation measures.
pub trait DeviationExt<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    /// Counts the number of indices at which the elements of the arrays `self`
    /// and `other` are equal.
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::EmptyInput` if `self` is empty
    /// * `ShapeMismatch` if `self` and `other` don't have the same shape
    fn count_eq(&self, other: &ArrayBase<S, D>) -> Result<usize, MultiInputError>
    where
        A: PartialEq;

    /// Counts the number of indices at which the elements of the arrays `self`
    /// and `other` are not equal.
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::EmptyInput` if `self` is empty
    /// * `ShapeMismatch` if `self` and `other` don't have the same shape
    fn count_neq(&self, other: &ArrayBase<S, D>) -> Result<usize, MultiInputError>
    where
        A: PartialEq;

    /// Computes the [squared L2 distance] between `self` and `other`.
    ///
    /// ```text
    ///  n
    ///  ∑  |aᵢ - bᵢ|²
    /// i=1
    /// ```
    ///
    /// where `self` is `a` and `other` is `b`.
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::EmptyInput` if `self` is empty
    /// * `ShapeMismatch` if `self` and `other` don't have the same shape
    ///
    /// [squared L2 distance]: https://en.wikipedia.org/wiki/Euclidean_distance#Squared_Euclidean_distance
    fn sq_l2_dist(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: AddAssign + Clone + Signed;

    /// Computes the [L2 distance] between `self` and `other`.
    ///
    /// ```text
    ///    n
    /// √  ∑  |aᵢ - bᵢ|²
    ///   i=1
    /// ```
    ///
    /// where `self` is `a` and `other` is `b`.
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::EmptyInput` if `self` is empty
    /// * `ShapeMismatch` if `self` and `other` don't have the same shape
    ///
    /// **Panics** if the type cast from `A` to `f64` fails.
    ///
    /// [L2 distance]: https://en.wikipedia.org/wiki/Euclidean_distance
    fn l2_dist(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive;

    /// Computes the [L1 distance] between `self` and `other`.
    ///
    /// ```text
    ///  n
    ///  ∑  |aᵢ - bᵢ|
    /// i=1
    /// ```
    ///
    /// where `self` is `a` and `other` is `b`.
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::EmptyInput` if `self` is empty
    /// * `ShapeMismatch` if `self` and `other` don't have the same shape
    ///
    /// [L1 distance]: https://en.wikipedia.org/wiki/Taxicab_geometry
    fn l1_dist(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: AddAssign + Clone + Signed;

    /// Computes the [L∞ distance] between `self` and `other`.
    ///
    /// ```text
    /// max(|aᵢ - bᵢ|)
    ///  ᵢ
    /// ```
    ///
    /// where `self` is `a` and `other` is `b`.
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::EmptyInput` if `self` is empty
    /// * `ShapeMismatch` if `self` and `other` don't have the same shape
    ///
    /// [L∞ distance]: https://en.wikipedia.org/wiki/Chebyshev_distance
    fn linf_dist(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: Clone + PartialOrd + Signed;

    /// Computes the [mean absolute error] between `self` and `other`.
    ///
    /// ```text
    ///        n
    /// 1/n *  ∑  |aᵢ - bᵢ|
    ///       i=1
    /// ```
    ///
    /// where `self` is `a` and `other` is `b`.
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::EmptyInput` if `self` is empty
    /// * `ShapeMismatch` if `self` and `other` don't have the same shape
    ///
    /// **Panics** if the type cast from `A` to `f64` fails.
    ///
    /// [mean absolute error]: https://en.wikipedia.org/wiki/Mean_absolute_error
    fn mean_abs_err(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive;

    /// Computes the [mean squared error] between `self` and `other`.
    ///
    /// ```text
    ///        n
    /// 1/n *  ∑  |aᵢ - bᵢ|²
    ///       i=1
    /// ```
    ///
    /// where `self` is `a` and `other` is `b`.
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::EmptyInput` if `self` is empty
    /// * `ShapeMismatch` if `self` and `other` don't have the same shape
    ///
    /// **Panics** if the type cast from `A` to `f64` fails.
    ///
    /// [mean squared error]: https://en.wikipedia.org/wiki/Mean_squared_error
    fn mean_sq_err(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive;

    /// Computes the unnormalized [root-mean-square error] between `self` and `other`.
    ///
    /// ```text
    /// √ mse(a, b)
    /// ```
    ///
    /// where `self` is `a` and `other` is `b`.
    ///
    /// where `mse` is the mean-squared-error.
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::EmptyInput` if `self` is empty
    /// * `ShapeMismatch` if `self` and `other` don't have the same shape
    ///
    /// **Panics** if the type cast from `A` to `f64` fails.
    ///
    /// [root-mean-square error]: https://en.wikipedia.org/wiki/Root-mean-square_deviation
    fn root_mean_sq_err(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive;

    /// Computes the [peak signal-to-noise ratio] between `self` and `other`.
    ///
    /// ```text
    /// 10 * log10(maxv^2 / mse(a, b))
    /// ```
    ///
    /// where `self` is `a`, `other` is `b`, `mse` is the mean-squared-error
    /// and `maxv` is the maximum possible value either array can take.
    ///
    /// The following **errors** may be returned:
    ///
    /// * `MultiInputError::EmptyInput` if `self` is empty
    /// * `ShapeMismatch` if `self` and `other` don't have the same shape
    ///
    /// **Panics** if the type cast from `A` to `f64` fails.
    ///
    /// [peak signal-to-noise ratio]: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    fn peak_signal_to_noise_ratio(
        &self,
        other: &ArrayBase<S, D>,
        maxv: A,
    ) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive;

    private_decl! {}
}

macro_rules! return_err_if_empty {
    ($arr:expr) => {
        if $arr.len() == 0 {
            return Err(MultiInputError::EmptyInput);
        }
    };
}
macro_rules! return_err_unless_same_shape {
    ($arr_a:expr, $arr_b:expr) => {
        if $arr_a.shape() != $arr_b.shape() {
            return Err(ShapeMismatch {
                first_shape: $arr_a.shape().to_vec(),
                second_shape: $arr_b.shape().to_vec(),
            }
            .into());
        }
    };
}

impl<A, S, D> DeviationExt<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn count_eq(&self, other: &ArrayBase<S, D>) -> Result<usize, MultiInputError>
    where
        A: PartialEq,
    {
        return_err_if_empty!(self);
        return_err_unless_same_shape!(self, other);

        let mut count = 0;

        Zip::from(self).and(other).apply(|a, b| {
            if a == b {
                count += 1;
            }
        });

        Ok(count)
    }

    fn count_neq(&self, other: &ArrayBase<S, D>) -> Result<usize, MultiInputError>
    where
        A: PartialEq,
    {
        self.count_eq(other).map(|n_eq| self.len() - n_eq)
    }

    fn sq_l2_dist(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: AddAssign + Clone + Signed,
    {
        return_err_if_empty!(self);
        return_err_unless_same_shape!(self, other);

        let mut result = A::zero();

        Zip::from(self).and(other).apply(|self_i, other_i| {
            let (a, b) = (self_i.clone(), other_i.clone());
            let abs_diff = (a - b).abs();
            result += abs_diff.clone() * abs_diff;
        });

        Ok(result)
    }

    fn l2_dist(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive,
    {
        let sq_l2_dist = self
            .sq_l2_dist(other)?
            .to_f64()
            .expect("failed cast from type A to f64");

        Ok(sq_l2_dist.sqrt())
    }

    fn l1_dist(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: AddAssign + Clone + Signed,
    {
        return_err_if_empty!(self);
        return_err_unless_same_shape!(self, other);

        let mut result = A::zero();

        Zip::from(self).and(other).apply(|self_i, other_i| {
            let (a, b) = (self_i.clone(), other_i.clone());
            result += (a - b).abs();
        });

        Ok(result)
    }

    fn linf_dist(&self, other: &ArrayBase<S, D>) -> Result<A, MultiInputError>
    where
        A: Clone + PartialOrd + Signed,
    {
        return_err_if_empty!(self);
        return_err_unless_same_shape!(self, other);

        let mut max = A::zero();

        Zip::from(self).and(other).apply(|self_i, other_i| {
            let (a, b) = (self_i.clone(), other_i.clone());
            let diff = (a - b).abs();
            if diff > max {
                max = diff;
            }
        });

        Ok(max)
    }

    fn mean_abs_err(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive,
    {
        let l1_dist = self
            .l1_dist(other)?
            .to_f64()
            .expect("failed cast from type A to f64");
        let n = self.len() as f64;

        Ok(l1_dist / n)
    }

    fn mean_sq_err(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive,
    {
        let sq_l2_dist = self
            .sq_l2_dist(other)?
            .to_f64()
            .expect("failed cast from type A to f64");
        let n = self.len() as f64;

        Ok(sq_l2_dist / n)
    }

    fn root_mean_sq_err(&self, other: &ArrayBase<S, D>) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive,
    {
        let msd = self.mean_sq_err(other)?;
        Ok(msd.sqrt())
    }

    fn peak_signal_to_noise_ratio(
        &self,
        other: &ArrayBase<S, D>,
        maxv: A,
    ) -> Result<f64, MultiInputError>
    where
        A: AddAssign + Clone + Signed + ToPrimitive,
    {
        let maxv_f = maxv.to_f64().expect("failed cast from type A to f64");
        let msd = self.mean_sq_err(&other)?;
        let psnr = 10. * f64::log10(maxv_f * maxv_f / msd);

        Ok(psnr)
    }

    private_impl! {}
}
