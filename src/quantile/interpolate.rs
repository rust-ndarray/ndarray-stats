//! Interpolation strategies.
use noisy_float::types::N64;
use num_traits::{Euclid, Float, FromPrimitive, NumOps, ToPrimitive};

use crate::maybe_nan::NotNone;

fn float_quantile_index(q: N64, len: usize) -> N64 {
    q * ((len - 1) as f64)
}

/// Returns the fraction that the quantile is between the lower and higher indices.
///
/// This ranges from 0, where the quantile exactly corresponds the lower index,
/// to 1, where the quantile exactly corresponds to the higher index.
fn float_quantile_index_fraction(q: N64, len: usize) -> N64 {
    float_quantile_index(q, len).fract()
}

/// Returns the index of the value on the lower side of the quantile.
pub(crate) fn lower_index(q: N64, len: usize) -> usize {
    float_quantile_index(q, len).floor().to_usize().unwrap()
}

/// Returns the index of the value on the higher side of the quantile.
pub(crate) fn higher_index(q: N64, len: usize) -> usize {
    float_quantile_index(q, len).ceil().to_usize().unwrap()
}

/// Used to provide an interpolation strategy to [`quantile_axis_mut`].
///
/// [`quantile_axis_mut`]: ../trait.QuantileExt.html#tymethod.quantile_axis_mut
pub trait Interpolate<T> {
    /// Returns `true` iff the lower value is needed to compute the
    /// interpolated value.
    #[doc(hidden)]
    fn needs_lower(q: N64, len: usize) -> bool;

    /// Returns `true` iff the higher value is needed to compute the
    /// interpolated value.
    #[doc(hidden)]
    fn needs_higher(q: N64, len: usize) -> bool;

    /// Computes the interpolated value.
    ///
    /// **Panics** if `None` is provided for the lower value when it's needed
    /// or if `None` is provided for the higher value when it's needed.
    #[doc(hidden)]
    fn interpolate(lower: Option<T>, higher: Option<T>, q: N64, len: usize) -> T;

    private_decl! {}
}

/// Select the higher value.
pub struct Higher;
/// Select the lower value.
pub struct Lower;
/// Select the nearest value.
pub struct Nearest;
/// Select the midpoint of the two values (`(lower + higher) / 2`).
pub struct Midpoint;
/// Linearly interpolate between the two values
/// (`lower + (higher - lower) * fraction`, where `fraction` is the
/// fractional part of the index surrounded by `lower` and `higher`).
pub struct Linear;

impl<T> Interpolate<T> for Higher {
    fn needs_lower(_q: N64, _len: usize) -> bool {
        false
    }
    fn needs_higher(_q: N64, _len: usize) -> bool {
        true
    }
    fn interpolate(_lower: Option<T>, higher: Option<T>, _q: N64, _len: usize) -> T {
        higher.unwrap()
    }
    private_impl! {}
}

impl<T> Interpolate<T> for Lower {
    fn needs_lower(_q: N64, _len: usize) -> bool {
        true
    }
    fn needs_higher(_q: N64, _len: usize) -> bool {
        false
    }
    fn interpolate(lower: Option<T>, _higher: Option<T>, _q: N64, _len: usize) -> T {
        lower.unwrap()
    }
    private_impl! {}
}

impl<T> Interpolate<T> for Nearest {
    fn needs_lower(q: N64, len: usize) -> bool {
        float_quantile_index_fraction(q, len) < 0.5
    }
    fn needs_higher(q: N64, len: usize) -> bool {
        !<Self as Interpolate<T>>::needs_lower(q, len)
    }
    fn interpolate(lower: Option<T>, higher: Option<T>, q: N64, len: usize) -> T {
        if <Self as Interpolate<T>>::needs_lower(q, len) {
            lower.unwrap()
        } else {
            higher.unwrap()
        }
    }
    private_impl! {}
}

macro_rules! impl_midpoint_interpolate_for_float {
    ($($t:ty),*) => {
        $(
            impl Interpolate<$t> for Midpoint {
                fn needs_lower(_q: N64, _len: usize) -> bool {
                    true
                }
                fn needs_higher(_q: N64, _len: usize) -> bool {
                    true
                }
                fn interpolate(lower: Option<$t>, higher: Option<$t>, _q: N64, _len: usize) -> $t {
                    let lower = lower.unwrap();
                    let higher = higher.unwrap();
                    lower + (higher - lower) / 2.0
                }
                private_impl! {}
            }
        )*
    }
}

impl_midpoint_interpolate_for_float!(f32, f64);

macro_rules! impl_midpoint_interpolate_for_integer {
    ($($t:ty),*) => {
        $(
            impl Interpolate<$t> for Midpoint {
                fn needs_lower(_q: N64, _len: usize) -> bool {
                    true
                }
                fn needs_higher(_q: N64, _len: usize) -> bool {
                    true
                }
                fn interpolate(lower: Option<$t>, higher: Option<$t>, _q: N64, _len: usize) -> $t {
                    let two = <$t>::from_u8(2).unwrap();
                    let (lower_half, lower_rem) = lower.unwrap().div_rem_euclid(&two);
                    let (higher_half, higher_rem) = higher.unwrap().div_rem_euclid(&two);
                    lower_half + higher_half + (lower_rem * higher_rem)
                }
                private_impl! {}
            }
        )*
    }
}

impl_midpoint_interpolate_for_integer!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);
impl_midpoint_interpolate_for_integer!(
    NotNone<i8>,
    NotNone<i16>,
    NotNone<i32>,
    NotNone<i64>,
    NotNone<i128>,
    NotNone<isize>,
    NotNone<u8>,
    NotNone<u16>,
    NotNone<u32>,
    NotNone<u64>,
    NotNone<u128>,
    NotNone<usize>
);

impl<T> Interpolate<T> for Linear
where
    T: NumOps + Clone + FromPrimitive + ToPrimitive,
{
    fn needs_lower(_q: N64, _len: usize) -> bool {
        true
    }
    fn needs_higher(_q: N64, _len: usize) -> bool {
        true
    }
    fn interpolate(lower: Option<T>, higher: Option<T>, q: N64, len: usize) -> T {
        let fraction = float_quantile_index_fraction(q, len).to_f64().unwrap();
        let lower = lower.unwrap();
        let higher = higher.unwrap();
        let lower_f64 = lower.to_f64().unwrap();
        let higher_f64 = higher.to_f64().unwrap();
        lower.clone() + T::from_f64(fraction * (higher_f64 - lower_f64)).unwrap()
    }
    private_impl! {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use noisy_float::types::n64;
    use quickcheck::TestResult;
    use quickcheck_macros::quickcheck;

    #[derive(Clone, Copy, Debug)]
    struct LowerHigherPair<T>(T, T);

    impl quickcheck::Arbitrary for LowerHigherPair<i64> {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            let (l, h) = loop {
                let (l, h) = (i64::arbitrary(g), i64::arbitrary(g));
                if l > h || h.checked_sub(l).is_none() {
                    continue;
                }
                break (l, h);
            };
            LowerHigherPair(l, h)
        }
    }

    impl From<LowerHigherPair<i64>> for (i64, i64) {
        fn from(value: LowerHigherPair<i64>) -> Self {
            (value.0, value.1)
        }
    }

    fn naive_midpoint_i64(lower: i64, higher: i64) -> i64 {
        // Overflows when higher is very big and lower is very small
        lower + (higher - lower) / 2
    }

    #[quickcheck]
    fn test_midpoint_algo_eq_naive_algo_i64(lh: LowerHigherPair<i64>) -> TestResult {
        let (lower, higher) = lh.into();
        let naive = naive_midpoint_i64(lower, higher);
        let midpoint = Midpoint::interpolate(Some(lower), Some(higher), n64(0.0), 0);
        TestResult::from_bool(naive == midpoint)
    }
}
