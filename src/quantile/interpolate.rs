//! Interpolation strategies.
use ndarray::azip;
use ndarray::prelude::*;
use noisy_float::types::N64;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::ops::{Add, Div};

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
    fn interpolate<D>(
        lower: Option<Array<T, D>>,
        higher: Option<Array<T, D>>,
        q: N64,
        len: usize,
    ) -> Array<T, D>
    where
        D: Dimension;
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
    fn interpolate<D>(
        _lower: Option<Array<T, D>>,
        higher: Option<Array<T, D>>,
        _q: N64,
        _len: usize,
    ) -> Array<T, D> {
        higher.unwrap()
    }
}

impl<T> Interpolate<T> for Lower {
    fn needs_lower(_q: N64, _len: usize) -> bool {
        true
    }
    fn needs_higher(_q: N64, _len: usize) -> bool {
        false
    }
    fn interpolate<D>(
        lower: Option<Array<T, D>>,
        _higher: Option<Array<T, D>>,
        _q: N64,
        _len: usize,
    ) -> Array<T, D> {
        lower.unwrap()
    }
}

impl<T> Interpolate<T> for Nearest {
    fn needs_lower(q: N64, len: usize) -> bool {
        float_quantile_index_fraction(q, len) < 0.5
    }
    fn needs_higher(q: N64, len: usize) -> bool {
        !<Self as Interpolate<T>>::needs_lower(q, len)
    }
    fn interpolate<D>(
        lower: Option<Array<T, D>>,
        higher: Option<Array<T, D>>,
        q: N64,
        len: usize,
    ) -> Array<T, D> {
        if <Self as Interpolate<T>>::needs_lower(q, len) {
            lower.unwrap()
        } else {
            higher.unwrap()
        }
    }
}

impl<T> Interpolate<T> for Midpoint
where
    T: Add<T, Output = T> + Div<T, Output = T> + Clone + FromPrimitive,
{
    fn needs_lower(_q: N64, _len: usize) -> bool {
        true
    }
    fn needs_higher(_q: N64, _len: usize) -> bool {
        true
    }
    fn interpolate<D>(
        lower: Option<Array<T, D>>,
        higher: Option<Array<T, D>>,
        _q: N64,
        _len: usize,
    ) -> Array<T, D>
    where
        D: Dimension,
    {
        let denom = T::from_u8(2).unwrap();
        (lower.unwrap() + higher.unwrap()).mapv_into(|x| x / denom.clone())
    }
}

impl<T> Interpolate<T> for Linear
where
    T: Add<T, Output = T> + Clone + FromPrimitive + ToPrimitive,
{
    fn needs_lower(_q: N64, _len: usize) -> bool {
        true
    }
    fn needs_higher(_q: N64, _len: usize) -> bool {
        true
    }
    fn interpolate<D>(
        lower: Option<Array<T, D>>,
        higher: Option<Array<T, D>>,
        q: N64,
        len: usize,
    ) -> Array<T, D>
    where
        D: Dimension,
    {
        let fraction = float_quantile_index_fraction(q, len).to_f64().unwrap();
        let mut a = lower.unwrap();
        let b = higher.unwrap();
        azip!(mut a, ref b in {
            let a_f64 = a.to_f64().unwrap();
            let b_f64 = b.to_f64().unwrap();
            *a = a.clone() + T::from_f64(fraction * (b_f64 - a_f64)).unwrap();
        });
        a
    }
}
