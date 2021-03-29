//! Information theory (e.g. entropy, KL divergence, etc.).
use crate::errors::{EmptyInput, MultiInputError, ShapeMismatch};
use ndarray::{Array, ArrayBase, Data, Dimension, Zip};
use num_traits::Float;

/// Extension trait for `ArrayBase` providing methods
/// to compute information theory quantities
/// (e.g. entropy, Kullback–Leibler divergence, etc.).
pub trait EntropyExt<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    /// Computes the [entropy] *S* of the array values, defined as
    ///
    /// ```text
    ///       n
    /// S = - ∑ xᵢ ln(xᵢ)
    ///      i=1
    /// ```
    ///
    /// If the array is empty, `Err(EmptyInput)` is returned.
    ///
    /// **Panics** if `ln` of any element in the array panics (which can occur for negative values for some `A`).
    ///
    /// ## Remarks
    ///
    /// The entropy is a measure used in [Information Theory]
    /// to describe a probability distribution: it only make sense
    /// when the array values sum to 1, with each entry between
    /// 0 and 1 (extremes included).
    ///
    /// The array values are **not** normalised by this function before
    /// computing the entropy to avoid introducing potentially
    /// unnecessary numerical errors (e.g. if the array were to be already normalised).
    ///
    /// By definition, *xᵢ ln(xᵢ)* is set to 0 if *xᵢ* is 0.
    ///
    /// [entropy]: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    /// [Information Theory]: https://en.wikipedia.org/wiki/Information_theory
    fn entropy(&self) -> Result<A, EmptyInput>
    where
        A: Float;

    /// Computes the [Kullback-Leibler divergence] *Dₖₗ(p,q)* between two arrays,
    /// where `self`=*p*.
    ///
    /// The Kullback-Leibler divergence is defined as:
    ///
    /// ```text
    ///              n
    /// Dₖₗ(p,q) = - ∑ pᵢ ln(qᵢ/pᵢ)
    ///             i=1
    /// ```
    ///
    /// If the arrays are empty, `Err(MultiInputError::EmptyInput)` is returned.
    /// If the array shapes are not identical,
    /// `Err(MultiInputError::ShapeMismatch)` is returned.
    ///
    /// **Panics** if, for a pair of elements *(pᵢ, qᵢ)* from *p* and *q*, computing
    /// *ln(qᵢ/pᵢ)* is a panic cause for `A`.
    ///
    /// ## Remarks
    ///
    /// The Kullback-Leibler divergence is a measure used in [Information Theory]
    /// to describe the relationship between two probability distribution: it only make sense
    /// when each array sums to 1 with entries between 0 and 1 (extremes included).
    ///
    /// The array values are **not** normalised by this function before
    /// computing the entropy to avoid introducing potentially
    /// unnecessary numerical errors (e.g. if the array were to be already normalised).
    ///
    /// By definition, *pᵢ ln(qᵢ/pᵢ)* is set to 0 if *pᵢ* is 0.
    ///
    /// [Kullback-Leibler divergence]: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    /// [Information Theory]: https://en.wikipedia.org/wiki/Information_theory
    fn kl_divergence<S2>(&self, q: &ArrayBase<S2, D>) -> Result<A, MultiInputError>
    where
        S2: Data<Elem = A>,
        A: Float;

    /// Computes the [cross entropy] *H(p,q)* between two arrays,
    /// where `self`=*p*.
    ///
    /// The cross entropy is defined as:
    ///
    /// ```text
    ///            n
    /// H(p,q) = - ∑ pᵢ ln(qᵢ)
    ///           i=1
    /// ```
    ///
    /// If the arrays are empty, `Err(MultiInputError::EmptyInput)` is returned.
    /// If the array shapes are not identical,
    /// `Err(MultiInputError::ShapeMismatch)` is returned.
    ///
    /// **Panics** if any element in *q* is negative and taking the logarithm of a negative number
    /// is a panic cause for `A`.
    ///
    /// ## Remarks
    ///
    /// The cross entropy is a measure used in [Information Theory]
    /// to describe the relationship between two probability distributions: it only makes sense
    /// when each array sums to 1 with entries between 0 and 1 (extremes included).
    ///
    /// The array values are **not** normalised by this function before
    /// computing the entropy to avoid introducing potentially
    /// unnecessary numerical errors (e.g. if the array were to be already normalised).
    ///
    /// The cross entropy is often used as an objective/loss function in
    /// [optimization problems], including [machine learning].
    ///
    /// By definition, *pᵢ ln(qᵢ)* is set to 0 if *pᵢ* is 0.
    ///
    /// [cross entropy]: https://en.wikipedia.org/wiki/Cross-entropy
    /// [Information Theory]: https://en.wikipedia.org/wiki/Information_theory
    /// [optimization problems]: https://en.wikipedia.org/wiki/Cross-entropy_method
    /// [machine learning]: https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression
    fn cross_entropy<S2>(&self, q: &ArrayBase<S2, D>) -> Result<A, MultiInputError>
    where
        S2: Data<Elem = A>,
        A: Float;

    private_decl! {}
}

impl<A, S, D> EntropyExt<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn entropy(&self) -> Result<A, EmptyInput>
    where
        A: Float,
    {
        if self.is_empty() {
            Err(EmptyInput)
        } else {
            let entropy = -self
                .mapv(|x| {
                    if x == A::zero() {
                        A::zero()
                    } else {
                        x * x.ln()
                    }
                })
                .sum();
            Ok(entropy)
        }
    }

    fn kl_divergence<S2>(&self, q: &ArrayBase<S2, D>) -> Result<A, MultiInputError>
    where
        A: Float,
        S2: Data<Elem = A>,
    {
        if self.is_empty() {
            return Err(MultiInputError::EmptyInput);
        }
        if self.shape() != q.shape() {
            return Err(ShapeMismatch {
                first_shape: self.shape().to_vec(),
                second_shape: q.shape().to_vec(),
            }
            .into());
        }

        let mut temp = Array::zeros(self.raw_dim());
        Zip::from(&mut temp)
            .and(self)
            .and(q)
            .for_each(|result, &p, &q| {
                *result = {
                    if p == A::zero() {
                        A::zero()
                    } else {
                        p * (q / p).ln()
                    }
                }
            });
        let kl_divergence = -temp.sum();
        Ok(kl_divergence)
    }

    fn cross_entropy<S2>(&self, q: &ArrayBase<S2, D>) -> Result<A, MultiInputError>
    where
        S2: Data<Elem = A>,
        A: Float,
    {
        if self.is_empty() {
            return Err(MultiInputError::EmptyInput);
        }
        if self.shape() != q.shape() {
            return Err(ShapeMismatch {
                first_shape: self.shape().to_vec(),
                second_shape: q.shape().to_vec(),
            }
            .into());
        }

        let mut temp = Array::zeros(self.raw_dim());
        Zip::from(&mut temp)
            .and(self)
            .and(q)
            .for_each(|result, &p, &q| {
                *result = {
                    if p == A::zero() {
                        A::zero()
                    } else {
                        p * q.ln()
                    }
                }
            });
        let cross_entropy = -temp.sum();
        Ok(cross_entropy)
    }

    private_impl! {}
}

#[cfg(test)]
mod tests {
    use super::EntropyExt;
    use crate::errors::{EmptyInput, MultiInputError};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};
    use noisy_float::types::n64;
    use std::f64;

    #[test]
    fn test_entropy_with_nan_values() {
        let a = array![f64::NAN, 1.];
        assert!(a.entropy().unwrap().is_nan());
    }

    #[test]
    fn test_entropy_with_empty_array_of_floats() {
        let a: Array1<f64> = array![];
        assert_eq!(a.entropy(), Err(EmptyInput));
    }

    #[test]
    fn test_entropy_with_array_of_floats() {
        // Array of probability values - normalized and positive.
        let a: Array1<f64> = array![
            0.03602474, 0.01900344, 0.03510129, 0.03414964, 0.00525311, 0.03368976, 0.00065396,
            0.02906146, 0.00063687, 0.01597306, 0.00787625, 0.00208243, 0.01450896, 0.01803418,
            0.02055336, 0.03029759, 0.03323628, 0.01218822, 0.0001873, 0.01734179, 0.03521668,
            0.02564429, 0.02421992, 0.03540229, 0.03497635, 0.03582331, 0.026558, 0.02460495,
            0.02437716, 0.01212838, 0.00058464, 0.00335236, 0.02146745, 0.00930306, 0.01821588,
            0.02381928, 0.02055073, 0.01483779, 0.02284741, 0.02251385, 0.00976694, 0.02864634,
            0.00802828, 0.03464088, 0.03557152, 0.01398894, 0.01831756, 0.0227171, 0.00736204,
            0.01866295,
        ];
        // Computed using scipy.stats.entropy
        let expected_entropy = 3.721606155686918;

        assert_abs_diff_eq!(a.entropy().unwrap(), expected_entropy, epsilon = 1e-6);
    }

    #[test]
    fn test_cross_entropy_and_kl_with_nan_values() -> Result<(), MultiInputError> {
        let a = array![f64::NAN, 1.];
        let b = array![2., 1.];
        assert!(a.cross_entropy(&b)?.is_nan());
        assert!(b.cross_entropy(&a)?.is_nan());
        assert!(a.kl_divergence(&b)?.is_nan());
        assert!(b.kl_divergence(&a)?.is_nan());
        Ok(())
    }

    #[test]
    fn test_cross_entropy_and_kl_with_same_n_dimension_but_different_n_elements() {
        let p = array![f64::NAN, 1.];
        let q = array![2., 1., 5.];
        assert!(q.cross_entropy(&p).is_err());
        assert!(p.cross_entropy(&q).is_err());
        assert!(q.kl_divergence(&p).is_err());
        assert!(p.kl_divergence(&q).is_err());
    }

    #[test]
    fn test_cross_entropy_and_kl_with_different_shape_but_same_n_elements() {
        // p: 3x2, 6 elements
        let p = array![[f64::NAN, 1.], [6., 7.], [10., 20.]];
        // q: 2x3, 6 elements
        let q = array![[2., 1., 5.], [1., 1., 7.],];
        assert!(q.cross_entropy(&p).is_err());
        assert!(p.cross_entropy(&q).is_err());
        assert!(q.kl_divergence(&p).is_err());
        assert!(p.kl_divergence(&q).is_err());
    }

    #[test]
    fn test_cross_entropy_and_kl_with_empty_array_of_floats() {
        let p: Array1<f64> = array![];
        let q: Array1<f64> = array![];
        assert!(p.cross_entropy(&q).unwrap_err().is_empty_input());
        assert!(p.kl_divergence(&q).unwrap_err().is_empty_input());
    }

    #[test]
    fn test_cross_entropy_and_kl_with_negative_qs() -> Result<(), MultiInputError> {
        let p = array![1.];
        let q = array![-1.];
        let cross_entropy: f64 = p.cross_entropy(&q)?;
        let kl_divergence: f64 = p.kl_divergence(&q)?;
        assert!(cross_entropy.is_nan());
        assert!(kl_divergence.is_nan());
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_cross_entropy_with_noisy_negative_qs() {
        let p = array![n64(1.)];
        let q = array![n64(-1.)];
        let _ = p.cross_entropy(&q);
    }

    #[test]
    #[should_panic]
    fn test_kl_with_noisy_negative_qs() {
        let p = array![n64(1.)];
        let q = array![n64(-1.)];
        let _ = p.kl_divergence(&q);
    }

    #[test]
    fn test_cross_entropy_and_kl_with_zeroes_p() -> Result<(), MultiInputError> {
        let p = array![0., 0.];
        let q = array![0., 0.5];
        assert_eq!(p.cross_entropy(&q)?, 0.);
        assert_eq!(p.kl_divergence(&q)?, 0.);
        Ok(())
    }

    #[test]
    fn test_cross_entropy_and_kl_with_zeroes_q_and_different_data_ownership(
    ) -> Result<(), MultiInputError> {
        let p = array![0.5, 0.5];
        let mut q = array![0.5, 0.];
        assert_eq!(p.cross_entropy(&q.view_mut())?, f64::INFINITY);
        assert_eq!(p.kl_divergence(&q.view_mut())?, f64::INFINITY);
        Ok(())
    }

    #[test]
    fn test_cross_entropy() -> Result<(), MultiInputError> {
        // Arrays of probability values - normalized and positive.
        let p: Array1<f64> = array![
            0.05340169, 0.02508511, 0.03460454, 0.00352313, 0.07837615, 0.05859495, 0.05782189,
            0.0471258, 0.05594036, 0.01630048, 0.07085162, 0.05365855, 0.01959158, 0.05020174,
            0.03801479, 0.00092234, 0.08515856, 0.00580683, 0.0156542, 0.0860375, 0.0724246,
            0.00727477, 0.01004402, 0.01854399, 0.03504082,
        ];
        let q: Array1<f64> = array![
            0.06622616, 0.0478948, 0.03227816, 0.06460884, 0.05795974, 0.01377489, 0.05604812,
            0.01202684, 0.01647579, 0.03392697, 0.01656126, 0.00867528, 0.0625685, 0.07381292,
            0.05489067, 0.01385491, 0.03639174, 0.00511611, 0.05700415, 0.05183825, 0.06703064,
            0.01813342, 0.0007763, 0.0735472, 0.05857833,
        ];
        // Computed using scipy.stats.entropy(p) + scipy.stats.entropy(p, q)
        let expected_cross_entropy = 3.385347705020779;

        assert_abs_diff_eq!(p.cross_entropy(&q)?, expected_cross_entropy, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_kl() -> Result<(), MultiInputError> {
        // Arrays of probability values - normalized and positive.
        let p: Array1<f64> = array![
            0.00150472, 0.01388706, 0.03495376, 0.03264211, 0.03067355, 0.02183501, 0.00137516,
            0.02213802, 0.02745017, 0.02163975, 0.0324602, 0.03622766, 0.00782343, 0.00222498,
            0.03028156, 0.02346124, 0.00071105, 0.00794496, 0.0127609, 0.02899124, 0.01281487,
            0.0230803, 0.01531864, 0.00518158, 0.02233383, 0.0220279, 0.03196097, 0.03710063,
            0.01817856, 0.03524661, 0.02902393, 0.00853364, 0.01255615, 0.03556958, 0.00400151,
            0.01335932, 0.01864965, 0.02371322, 0.02026543, 0.0035375, 0.01988341, 0.02621831,
            0.03564644, 0.01389121, 0.03151622, 0.03195532, 0.00717521, 0.03547256, 0.00371394,
            0.01108706,
        ];
        let q: Array1<f64> = array![
            0.02038386, 0.03143914, 0.02630206, 0.0171595, 0.0067072, 0.00911324, 0.02635717,
            0.01269113, 0.0302361, 0.02243133, 0.01902902, 0.01297185, 0.02118908, 0.03309548,
            0.01266687, 0.0184529, 0.01830936, 0.03430437, 0.02898924, 0.02238251, 0.0139771,
            0.01879774, 0.02396583, 0.03019978, 0.01421278, 0.02078981, 0.03542451, 0.02887438,
            0.01261783, 0.01014241, 0.03263407, 0.0095969, 0.01923903, 0.0051315, 0.00924686,
            0.00148845, 0.00341391, 0.01480373, 0.01920798, 0.03519871, 0.03315135, 0.02099325,
            0.03251755, 0.00337555, 0.03432165, 0.01763753, 0.02038337, 0.01923023, 0.01438769,
            0.02082707,
        ];
        // Computed using scipy.stats.entropy(p, q)
        let expected_kl = 0.3555862567800096;

        assert_abs_diff_eq!(p.kl_divergence(&q)?, expected_kl, epsilon = 1e-6);
        Ok(())
    }
}
