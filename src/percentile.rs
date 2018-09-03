use interpolate::Interpolate;
use ndarray::prelude::*;
use ndarray::{Data, DataMut, RemoveAxis};
use sort::Sort1dExt;

/// Interpolation strategies.
pub mod interpolate {
    use ndarray::prelude::*;
    use num_traits::FromPrimitive;
    use std::ops::{Add, Div, Mul, Sub};

    /// Used to provide an interpolation strategy to [`percentile_axis_mut`].
    ///
    /// [`percentile_axis_mut`]: ../trait.PercentileExt.html#tymethod.percentile_axis_mut
    pub trait Interpolate<T> {
        #[doc(hidden)]
        fn float_percentile_index(q: f64, len: usize) -> f64 {
            ((len - 1) as f64) * q
        }
        #[doc(hidden)]
        fn lower_index(q: f64, len: usize) -> usize {
            Self::float_percentile_index(q, len).floor() as usize
        }
        #[doc(hidden)]
        fn upper_index(q: f64, len: usize) -> usize {
            Self::float_percentile_index(q, len).ceil() as usize
        }
        #[doc(hidden)]
        fn float_percentile_index_fraction(q: f64, len: usize) -> f64 {
            Self::float_percentile_index(q, len) - (Self::lower_index(q, len) as f64)
        }
        #[doc(hidden)]
        fn needs_lower(q: f64, len: usize) -> bool;
        #[doc(hidden)]
        fn needs_upper(q: f64, len: usize) -> bool;
        #[doc(hidden)]
        fn interpolate<D>(
            lower: Option<Array<T, D>>,
            upper: Option<Array<T, D>>,
            q: f64,
            len: usize,
        ) -> Array<T, D>
        where
            D: Dimension;
    }

    /// Select the upper value.
    pub struct Upper;
    /// Select the lower value.
    pub struct Lower;
    /// Select the nearest value.
    pub struct Nearest;
    /// Select the midpoint of the two values.
    pub struct Midpoint;
    /// Linearly interpolate between the two values.
    pub struct Linear;

    impl<T> Interpolate<T> for Upper {
        fn needs_lower(_q: f64, _len: usize) -> bool {
            false
        }
        fn needs_upper(_q: f64, _len: usize) -> bool {
            true
        }
        fn interpolate<D>(
            _lower: Option<Array<T, D>>,
            upper: Option<Array<T, D>>,
            _q: f64,
            _len: usize,
        ) -> Array<T, D> {
            upper.unwrap()
        }
    }

    impl<T> Interpolate<T> for Lower {
        fn needs_lower(_q: f64, _len: usize) -> bool {
            true
        }
        fn needs_upper(_q: f64, _len: usize) -> bool {
            false
        }
        fn interpolate<D>(
            lower: Option<Array<T, D>>,
            _upper: Option<Array<T, D>>,
            _q: f64,
            _len: usize,
        ) -> Array<T, D> {
            lower.unwrap()
        }
    }

    impl<T> Interpolate<T> for Nearest {
        fn needs_lower(q: f64, len: usize) -> bool {
            let lower = <Self as Interpolate<T>>::lower_index(q, len);
            ((lower as f64) - <Self as Interpolate<T>>::float_percentile_index(q, len)) <= 0.
        }
        fn needs_upper(q: f64, len: usize) -> bool {
            !<Self as Interpolate<T>>::needs_lower(q, len)
        }
        fn interpolate<D>(
            lower: Option<Array<T, D>>,
            upper: Option<Array<T, D>>,
            q: f64,
            len: usize,
        ) -> Array<T, D> {
            if <Self as Interpolate<T>>::needs_lower(q, len) {
                lower.unwrap()
            } else {
                upper.unwrap()
            }
        }
    }

    impl<T> Interpolate<T> for Midpoint
    where
        T: Add<T, Output = T> + Div<T, Output = T> + Clone + FromPrimitive,
    {
        fn needs_lower(_q: f64, _len: usize) -> bool {
            true
        }
        fn needs_upper(_q: f64, _len: usize) -> bool {
            true
        }
        fn interpolate<D>(
            lower: Option<Array<T, D>>,
            upper: Option<Array<T, D>>,
            _q: f64,
            _len: usize,
        ) -> Array<T, D>
        where
            D: Dimension,
        {
            let denom = T::from_u8(2).unwrap();
            (lower.unwrap() + upper.unwrap()).mapv_into(|x| x / denom.clone())
        }
    }

    impl<T> Interpolate<T> for Linear
    where
        T: Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Clone + FromPrimitive,
    {
        fn needs_lower(_q: f64, _len: usize) -> bool {
            true
        }
        fn needs_upper(_q: f64, _len: usize) -> bool {
            true
        }
        fn interpolate<D>(
            lower: Option<Array<T, D>>,
            upper: Option<Array<T, D>>,
            q: f64,
            len: usize,
        ) -> Array<T, D>
        where
            D: Dimension,
        {
            let fraction = T::from_f64(<Self as Interpolate<T>>::float_percentile_index_fraction(
                q, len,
            )).unwrap();
            let a = lower.unwrap().mapv_into(|x| x * fraction.clone());
            let b = upper
                .unwrap()
                .mapv_into(|x| x * (T::from_u8(1).unwrap() - fraction.clone()));
            a + b
        }
    }
}

/// Percentile methods.
pub trait PercentileExt<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    /// Return the qth percentile of the data along the specified axis.
    ///
    /// `q` needs to be a float between 0 and 1, bounds included.
    /// The qth percentile for a 1-dimensional lane of length `N` is defined
    /// as the element that would be indexed as `(N-1)q` if the lane were to be sorted
    /// in increasing order.
    /// If `(N-1)q` is not an integer the desired percentile lies between
    /// two data points: we return the lower, nearest, higher or interpolated
    /// value depending on the type `Interpolate` bound `I`.
    ///
    /// Some examples:
    /// - `q=0.` returns the minimum along each 1-dimensional lane;
    /// - `q=0.5` returns the median along each 1-dimensional lane;
    /// - `q=1.` returns the maximum along each 1-dimensional lane.
    /// (`q=0` and `q=1` are considered improper percentiles)
    ///
    /// The array is shuffled **in place** along each 1-dimensional lane in
    /// order to produce the required percentile without allocating a copy
    /// of the original array. Each 1-dimensional lane is shuffled independently
    /// from the others.
    /// No assumptions should be made on the ordering of the array elements
    /// after this computation.
    ///
    /// Complexity ([quickselect](https://en.wikipedia.org/wiki/Quickselect)):
    /// - average case: O(`m`);
    /// - worst case: O(`m`^2);
    /// where `m` is the number of elements in the array.
    ///
    /// **Panics** if `axis` is out of bounds or if `q` is not between
    /// `0.` and `1.` (inclusive).
    fn percentile_axis_mut<I>(&mut self, axis: Axis, q: f64) -> Array<A, D::Smaller>
    where
        D: RemoveAxis,
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>;
}

impl<A, S, D> PercentileExt<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn percentile_axis_mut<I>(&mut self, axis: Axis, q: f64) -> Array<A, D::Smaller>
    where
        D: RemoveAxis,
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>,
    {
        assert!((0. <= q) && (q <= 1.));
        let mut lower = None;
        let mut upper = None;
        let axis_len = self.len_of(axis);
        if I::needs_lower(q, axis_len) {
            let lower_index = I::lower_index(q, axis_len);
            lower = Some(self.map_axis_mut(axis, |mut x| x.sorted_get_mut(lower_index)));
            if I::needs_upper(q, axis_len) {
                let upper_index = I::upper_index(q, axis_len);
                let relative_upper_index = upper_index - lower_index;
                upper = Some(self.map_axis_mut(axis, |mut x| {
                    x.slice_mut(s![lower_index..])
                        .sorted_get_mut(relative_upper_index)
                }));
            };
        } else {
            upper = Some(
                self.map_axis_mut(axis, |mut x| x.sorted_get_mut(I::upper_index(q, axis_len))),
            );
        };
        I::interpolate(lower, upper, q, axis_len)
    }
}
