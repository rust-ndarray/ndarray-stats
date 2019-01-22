//! Summary statistics (e.g. mean, variance, etc.).
use ndarray::{ArrayBase, Data, Dimension};
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
            Some(entropy)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EntropyExt;
    use std::f64;
    use approx::abs_diff_eq;
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
        let a: Array1<f64> = array![
            0.70850547, 0.32496524, 0.4512601 , 0.19634812, 0.52430767,
            0.77200268, 0.30947147, 0.01089479, 0.04280482, 0.18548377,
            0.7886273 , 0.23487162, 0.54353668, 0.43455954, 0.8224537 ,
            0.60031256, 0.69876954, 0.95906628, 0.20305543, 0.85397668,
            0.50892232, 0.65533253, 0.64384601, 0.86091271, 0.31692328,
            0.45576697, 0.66077109, 0.23469551, 0.42808089, 0.20234666,
            0.14972765, 0.34240363, 0.59198436, 0.05764641, 0.10238259,
            0.06544647, 0.74466137, 0.58182716, 0.5583189 , 0.36093108,
            0.60681015, 0.45062613, 0.83282631, 0.77114486, 0.35229367,
            0.36383337, 0.78485847, 0.56853643, 0.80326787, 0.04409981,
        ];
        // Computed using scipy.stats.entropy
        let expected_entropy = 3.7371557453896727;

        abs_diff_eq!(a.entropy().unwrap(), expected_entropy, epsilon = f64::EPSILON);
    }
}
