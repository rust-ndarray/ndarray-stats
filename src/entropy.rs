//! Summary statistics (e.g. mean, variance, etc.).
use ndarray::{Array1, ArrayBase, Data, Dimension};
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
    /// If the array is empty, `None` is returned.
    ///
    /// **Panics** if any element in the array is negative.
    ///
    /// ## Remarks
    ///
    /// The entropy is a measure used in [Information Theory]
    /// to describe a probability distribution: it only make sense
    /// when the array values sum to 1, with each entry between
    /// 0 and 1 (extremes included).
    ///
    /// By definition, *xᵢ ln(xᵢ)* is set to 0 if *xᵢ* is 0.
    ///
    /// [entropy]: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    /// [Information Theory]: https://en.wikipedia.org/wiki/Information_theory
    fn entropy(&self) -> Option<A>
    where
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
    /// If the arrays are empty or their lengths are not equal, `None` is returned.
    ///
    /// **Panics** if any element in *q* is negative.
    ///
    /// ## Remarks
    ///
    /// The cross entropy is a measure used in [Information Theory]
    /// to describe the relationship between two probability distribution: it only make sense
    /// when each array sums to 1 with entries between 0 and 1 (extremes included).
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
    fn cross_entropy(&self, q: &Self) -> Option<A>
    where
        A: Float;
}


impl<A, S, D> EntropyExt<A, S, D> for ArrayBase<S, D>
    where
        S: Data<Elem = A>,
        D: Dimension,
{
    fn entropy(&self) -> Option<A>
        where
            A: Float
    {
        if self.len() == 0 {
            None
        } else {
            let entropy = self.map(
                |x| {
                    if *x == A::zero() {
                        A::zero()
                    } else {
                        *x * x.ln()
                    }
                }
            ).sum();
            Some(-entropy)
        }
    }

    fn cross_entropy(&self, q: &Self) -> Option<A>
        where
            A: Float
    {
        if (self.len() == 0) | (self.len() != q.len()) {
            None
        } else {
            let cross_entropy: A = self.iter().zip(q.iter()).map(
                |(p, q)| {
                    if *p == A::zero() {
                        A::zero()
                    } else {
                        *p * q.ln()
                    }
                }
            ).collect::<Array1<A>>().sum();
            Some(-cross_entropy)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EntropyExt;
    use std::f64;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1};

    #[test]
    fn test_entropy_with_nan_values() {
        let a = array![f64::NAN, 1.];
        assert!(a.entropy().unwrap().is_nan());
    }

    #[test]
    fn test_entropy_with_empty_array_of_floats() {
        let a: Array1<f64> = array![];
        assert!(a.entropy().is_none());
    }

    #[test]
    fn test_entropy_with_array_of_floats() {
        // Array of probability values - normalized and positive.
        let a: Array1<f64> = array![
            0.03602474, 0.01900344, 0.03510129, 0.03414964, 0.00525311,
            0.03368976, 0.00065396, 0.02906146, 0.00063687, 0.01597306,
            0.00787625, 0.00208243, 0.01450896, 0.01803418, 0.02055336,
            0.03029759, 0.03323628, 0.01218822, 0.0001873 , 0.01734179,
            0.03521668, 0.02564429, 0.02421992, 0.03540229, 0.03497635,
            0.03582331, 0.026558  , 0.02460495, 0.02437716, 0.01212838,
            0.00058464, 0.00335236, 0.02146745, 0.00930306, 0.01821588,
            0.02381928, 0.02055073, 0.01483779, 0.02284741, 0.02251385,
            0.00976694, 0.02864634, 0.00802828, 0.03464088, 0.03557152,
            0.01398894, 0.01831756, 0.0227171 , 0.00736204, 0.01866295,
        ];
        // Computed using scipy.stats.entropy
        let expected_entropy = 3.721606155686918;

        assert_abs_diff_eq!(a.entropy().unwrap(), expected_entropy, epsilon = 1e-6);
    }
}
