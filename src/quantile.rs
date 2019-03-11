use interpolate::Interpolate;
use ndarray::prelude::*;
use ndarray::{s, Data, DataMut, RemoveAxis};
use std::cmp;
use {MaybeNan, MaybeNanExt, Sort1dExt};

/// Interpolation strategies.
pub mod interpolate {
    use ndarray::azip;
    use ndarray::prelude::*;
    use num_traits::{FromPrimitive, NumOps, ToPrimitive};

    /// Used to provide an interpolation strategy to [`quantile_axis_mut`].
    ///
    /// [`quantile_axis_mut`]: ../trait.QuantileExt.html#tymethod.quantile_axis_mut
    pub trait Interpolate<T> {
        #[doc(hidden)]
        fn float_quantile_index(q: f64, len: usize) -> f64 {
            ((len - 1) as f64) * q
        }
        #[doc(hidden)]
        fn lower_index(q: f64, len: usize) -> usize {
            Self::float_quantile_index(q, len).floor() as usize
        }
        #[doc(hidden)]
        fn higher_index(q: f64, len: usize) -> usize {
            Self::float_quantile_index(q, len).ceil() as usize
        }
        #[doc(hidden)]
        fn float_quantile_index_fraction(q: f64, len: usize) -> f64 {
            Self::float_quantile_index(q, len).fract()
        }
        #[doc(hidden)]
        fn needs_lower(q: f64, len: usize) -> bool;
        #[doc(hidden)]
        fn needs_higher(q: f64, len: usize) -> bool;
        #[doc(hidden)]
        fn interpolate<D>(
            lower: Option<Array<T, D>>,
            higher: Option<Array<T, D>>,
            q: f64,
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
        fn needs_lower(_q: f64, _len: usize) -> bool {
            false
        }
        fn needs_higher(_q: f64, _len: usize) -> bool {
            true
        }
        fn interpolate<D>(
            _lower: Option<Array<T, D>>,
            higher: Option<Array<T, D>>,
            _q: f64,
            _len: usize,
        ) -> Array<T, D> {
            higher.unwrap()
        }
    }

    impl<T> Interpolate<T> for Lower {
        fn needs_lower(_q: f64, _len: usize) -> bool {
            true
        }
        fn needs_higher(_q: f64, _len: usize) -> bool {
            false
        }
        fn interpolate<D>(
            lower: Option<Array<T, D>>,
            _higher: Option<Array<T, D>>,
            _q: f64,
            _len: usize,
        ) -> Array<T, D> {
            lower.unwrap()
        }
    }

    impl<T> Interpolate<T> for Nearest {
        fn needs_lower(q: f64, len: usize) -> bool {
            <Self as Interpolate<T>>::float_quantile_index_fraction(q, len) < 0.5
        }
        fn needs_higher(q: f64, len: usize) -> bool {
            !<Self as Interpolate<T>>::needs_lower(q, len)
        }
        fn interpolate<D>(
            lower: Option<Array<T, D>>,
            higher: Option<Array<T, D>>,
            q: f64,
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
        T: NumOps + Clone + FromPrimitive,
    {
        fn needs_lower(_q: f64, _len: usize) -> bool {
            true
        }
        fn needs_higher(_q: f64, _len: usize) -> bool {
            true
        }
        fn interpolate<D>(
            lower: Option<Array<T, D>>,
            higher: Option<Array<T, D>>,
            _q: f64,
            _len: usize,
        ) -> Array<T, D>
        where
            D: Dimension,
        {
            let denom = T::from_u8(2).unwrap();
            let mut lower = lower.unwrap();
            let higher = higher.unwrap();
            azip!(
                mut lower, ref higher in {
                    *lower = lower.clone() + (higher.clone() - lower.clone()) / denom.clone()
                }
            );
            lower
        }
    }

    impl<T> Interpolate<T> for Linear
    where
        T: NumOps + Clone + FromPrimitive + ToPrimitive,
    {
        fn needs_lower(_q: f64, _len: usize) -> bool {
            true
        }
        fn needs_higher(_q: f64, _len: usize) -> bool {
            true
        }
        fn interpolate<D>(
            lower: Option<Array<T, D>>,
            higher: Option<Array<T, D>>,
            q: f64,
            len: usize,
        ) -> Array<T, D>
        where
            D: Dimension,
        {
            let fraction = <Self as Interpolate<T>>::float_quantile_index_fraction(q, len);
            let mut a = lower.unwrap();
            let b = higher.unwrap();
            azip!(mut a, ref b in {
                let a_f64 = a.to_f64().unwrap();
                let b_f64 = b.to_f64().unwrap();
                *a = a.clone() + T::from_f64((b_f64 - a_f64) * fraction).unwrap();
            });
            a
        }
    }
}

/// Quantile methods for `ArrayBase`.
pub trait QuantileExt<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    /// Finds the index of the minimum value of the array.
    ///
    /// Returns `None` if any of the pairwise orderings tested by the function
    /// are undefined. (For example, this occurs if there are any
    /// floating-point NaN values in the array.)
    ///
    /// Returns `None` if the array is empty.
    ///
    /// Even if there are multiple (equal) elements that are minima, only one
    /// index is returned. (Which one is returned is unspecified and may depend
    /// on the memory layout of the array.)
    ///
    /// # Example
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate ndarray_stats;
    ///
    /// use ndarray::array;
    /// use ndarray_stats::QuantileExt;
    ///
    /// let a = array![[1., 3., 5.],
    ///                [2., 0., 6.]];
    /// assert_eq!(a.argmin(), Some((1, 1)));
    /// ```
    fn argmin(&self) -> Option<D::Pattern>
    where
        A: PartialOrd;

    /// Finds the first index of the minimum value of the array ignoring nan values.
    ///
    /// Returns `None` if the array is empty.
    ///
    /// **Warning** This method will return a None value if none of the values
    /// in the array are non-NaN values.
    ///
    /// # Example
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate ndarray_stats;
    ///
    /// use ndarray::array;
    /// use ndarray_stats::QuantileExt;
    ///
    /// let a = array![[::std::f64::NAN, 3., 5.],
    ///                [2., 0., 6.]];
    /// assert_eq!(a.argmin_skipnan(), Some((1, 1)));
    /// ```
    fn argmin_skipnan(&self) -> Option<D::Pattern>
    where
        A: MaybeNan,
        A::NotNan: Ord + std::fmt::Debug;

    /// Finds the elementwise minimum of the array.
    ///
    /// Returns `None` if any of the pairwise orderings tested by the function
    /// are undefined. (For example, this occurs if there are any
    /// floating-point NaN values in the array.)
    ///
    /// Additionally, returns `None` if the array is empty.
    ///
    /// Even if there are multiple (equal) elements that are minima, only one
    /// is returned. (Which one is returned is unspecified and may depend on
    /// the memory layout of the array.)
    fn min(&self) -> Option<&A>
    where
        A: PartialOrd;

    /// Finds the elementwise minimum of the array, skipping NaN values.
    ///
    /// Even if there are multiple (equal) elements that are minima, only one
    /// is returned. (Which one is returned is unspecified and may depend on
    /// the memory layout of the array.)
    ///
    /// **Warning** This method will return a NaN value if none of the values
    /// in the array are non-NaN values. Note that the NaN value might not be
    /// in the array.
    fn min_skipnan(&self) -> &A
    where
        A: MaybeNan,
        A::NotNan: Ord;

    /// Finds the index of the maximum value of the array.
    ///
    /// Returns `None` if any of the pairwise orderings tested by the function
    /// are undefined. (For example, this occurs if there are any
    /// floating-point NaN values in the array.)
    ///
    /// Returns `None` if the array is empty.
    ///
    /// Even if there are multiple (equal) elements that are maxima, only one
    /// index is returned. (Which one is returned is unspecified and may depend
    /// on the memory layout of the array.)
    ///
    /// # Example
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate ndarray_stats;
    ///
    /// use ndarray::array;
    /// use ndarray_stats::QuantileExt;
    ///
    /// let a = array![[1., 3., 7.],
    ///                [2., 5., 6.]];
    /// assert_eq!(a.argmax(), Some((0, 2)));
    /// ```
    fn argmax(&self) -> Option<D::Pattern>
    where
        A: PartialOrd;

    /// Finds the elementwise maximum of the array.
    ///
    /// Returns `None` if any of the pairwise orderings tested by the function
    /// are undefined. (For example, this occurs if there are any
    /// floating-point NaN values in the array.)
    ///
    /// Additionally, returns `None` if the array is empty.
    ///
    /// Even if there are multiple (equal) elements that are maxima, only one
    /// is returned. (Which one is returned is unspecified and may depend on
    /// the memory layout of the array.)
    fn max(&self) -> Option<&A>
    where
        A: PartialOrd;

    /// Finds the elementwise maximum of the array, skipping NaN values.
    ///
    /// Even if there are multiple (equal) elements that are maxima, only one
    /// is returned. (Which one is returned is unspecified and may depend on
    /// the memory layout of the array.)
    ///
    /// **Warning** This method will return a NaN value if none of the values
    /// in the array are non-NaN values. Note that the NaN value might not be
    /// in the array.
    fn max_skipnan(&self) -> &A
    where
        A: MaybeNan,
        A::NotNan: Ord;

    /// Return the qth quantile of the data along the specified axis.
    ///
    /// `q` needs to be a float between 0 and 1, bounds included.
    /// The qth quantile for a 1-dimensional lane of length `N` is defined
    /// as the element that would be indexed as `(N-1)q` if the lane were to be sorted
    /// in increasing order.
    /// If `(N-1)q` is not an integer the desired quantile lies between
    /// two data points: we return the lower, nearest, higher or interpolated
    /// value depending on the type `Interpolate` bound `I`.
    ///
    /// Some examples:
    /// - `q=0.` returns the minimum along each 1-dimensional lane;
    /// - `q=0.5` returns the median along each 1-dimensional lane;
    /// - `q=1.` returns the maximum along each 1-dimensional lane.
    /// (`q=0` and `q=1` are considered improper quantiles)
    ///
    /// The array is shuffled **in place** along each 1-dimensional lane in
    /// order to produce the required quantile without allocating a copy
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
    /// **Panics** if `axis` is out of bounds, if the axis has length 0, or if
    /// `q` is not between `0.` and `1.` (inclusive).
    fn quantile_axis_mut<I>(&mut self, axis: Axis, q: f64) -> Array<A, D::Smaller>
    where
        D: RemoveAxis,
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>;

    /// Return the `q`th quantile of the data along the specified axis, skipping NaN values.
    ///
    /// See [`quantile_axis_mut`](##tymethod.quantile_axis_mut) for details.
    fn quantile_axis_skipnan_mut<I>(&mut self, axis: Axis, q: f64) -> Array<A, D::Smaller>
    where
        D: RemoveAxis,
        A: MaybeNan,
        A::NotNan: Clone + Ord,
        S: DataMut,
        I: Interpolate<A::NotNan>;
}

impl<A, S, D> QuantileExt<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn argmin(&self) -> Option<D::Pattern>
    where
        A: PartialOrd,
    {
        let mut current_min = self.first()?;
        let mut current_pattern_min = D::zeros(self.ndim()).into_pattern();

        for (pattern, elem) in self.indexed_iter() {
            if elem.partial_cmp(current_min)? == cmp::Ordering::Less {
                current_pattern_min = pattern;
                current_min = elem
            }
        }

        Some(current_pattern_min)
    }

    fn argmin_skipnan(&self) -> Option<D::Pattern>
    where
        A: MaybeNan,
        A::NotNan: Ord + std::fmt::Debug,
    {
        let mut current_min = self.first().and_then(|v| v.try_as_not_nan());
        let mut current_pattern_min = D::zeros(self.ndim()).into_pattern();
        for (pattern, elem) in self.indexed_iter() {
            let elem_not_nan = elem.try_as_not_nan();
            if elem_not_nan.is_some()
                && (current_min.is_none()
                    || elem_not_nan.partial_cmp(&current_min) == Some(cmp::Ordering::Less))
            {
                current_pattern_min = pattern;
                current_min = elem_not_nan;
            }
        }
        current_min.map({ |_| current_pattern_min })
    }

    fn min(&self) -> Option<&A>
    where
        A: PartialOrd,
    {
        let first = self.first()?;
        self.fold(Some(first), |acc, elem| match elem.partial_cmp(acc?)? {
            cmp::Ordering::Less => Some(elem),
            _ => acc,
        })
    }

    fn min_skipnan(&self) -> &A
    where
        A: MaybeNan,
        A::NotNan: Ord,
    {
        let first = self.first().and_then(|v| v.try_as_not_nan());
        A::from_not_nan_ref_opt(self.fold_skipnan(first, |acc, elem| {
            Some(match acc {
                Some(acc) => acc.min(elem),
                None => elem,
            })
        }))
    }

    fn argmax(&self) -> Option<D::Pattern>
    where
        A: PartialOrd,
    {
        let mut current_max = self.first()?;
        let mut current_pattern_max = D::zeros(self.ndim()).into_pattern();

        for (pattern, elem) in self.indexed_iter() {
            if elem.partial_cmp(current_max)? == cmp::Ordering::Greater {
                current_pattern_max = pattern;
                current_max = elem
            }
        }

        Some(current_pattern_max)
    }

    fn max(&self) -> Option<&A>
    where
        A: PartialOrd,
    {
        let first = self.first()?;
        self.fold(Some(first), |acc, elem| match elem.partial_cmp(acc?)? {
            cmp::Ordering::Greater => Some(elem),
            _ => acc,
        })
    }

    fn max_skipnan(&self) -> &A
    where
        A: MaybeNan,
        A::NotNan: Ord,
    {
        let first = self.first().and_then(|v| v.try_as_not_nan());
        A::from_not_nan_ref_opt(self.fold_skipnan(first, |acc, elem| {
            Some(match acc {
                Some(acc) => acc.max(elem),
                None => elem,
            })
        }))
    }

    fn quantile_axis_mut<I>(&mut self, axis: Axis, q: f64) -> Array<A, D::Smaller>
    where
        D: RemoveAxis,
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>,
    {
        assert!((0. <= q) && (q <= 1.));
        let mut lower = None;
        let mut higher = None;
        let axis_len = self.len_of(axis);
        if I::needs_lower(q, axis_len) {
            let lower_index = I::lower_index(q, axis_len);
            lower = Some(self.map_axis_mut(axis, |mut x| x.sorted_get_mut(lower_index)));
            if I::needs_higher(q, axis_len) {
                let higher_index = I::higher_index(q, axis_len);
                let relative_higher_index = higher_index - lower_index;
                higher = Some(self.map_axis_mut(axis, |mut x| {
                    x.slice_mut(s![lower_index..])
                        .sorted_get_mut(relative_higher_index)
                }));
            };
        } else {
            higher = Some(
                self.map_axis_mut(axis, |mut x| x.sorted_get_mut(I::higher_index(q, axis_len))),
            );
        };
        I::interpolate(lower, higher, q, axis_len)
    }

    fn quantile_axis_skipnan_mut<I>(&mut self, axis: Axis, q: f64) -> Array<A, D::Smaller>
    where
        D: RemoveAxis,
        A: MaybeNan,
        A::NotNan: Clone + Ord,
        S: DataMut,
        I: Interpolate<A::NotNan>,
    {
        self.map_axis_mut(axis, |lane| {
            let mut not_nan = A::remove_nan_mut(lane);
            A::from_not_nan_opt(if not_nan.is_empty() {
                None
            } else {
                Some(
                    not_nan
                        .quantile_axis_mut::<I>(Axis(0), q)
                        .into_raw_vec()
                        .remove(0),
                )
            })
        })
    }
}

/// Quantile methods for 1-D arrays.
pub trait Quantile1dExt<A, S>
where
    S: Data<Elem = A>,
{
    /// Return the qth quantile of the data.
    ///
    /// `q` needs to be a float between 0 and 1, bounds included.
    /// The qth quantile for a 1-dimensional array of length `N` is defined
    /// as the element that would be indexed as `(N-1)q` if the array were to be sorted
    /// in increasing order.
    /// If `(N-1)q` is not an integer the desired quantile lies between
    /// two data points: we return the lower, nearest, higher or interpolated
    /// value depending on the type `Interpolate` bound `I`.
    ///
    /// Some examples:
    /// - `q=0.` returns the minimum;
    /// - `q=0.5` returns the median;
    /// - `q=1.` returns the maximum.
    /// (`q=0` and `q=1` are considered improper quantiles)
    ///
    /// The array is shuffled **in place** in order to produce the required quantile
    /// without allocating a copy.
    /// No assumptions should be made on the ordering of the array elements
    /// after this computation.
    ///
    /// Complexity ([quickselect](https://en.wikipedia.org/wiki/Quickselect)):
    /// - average case: O(`m`);
    /// - worst case: O(`m`^2);
    /// where `m` is the number of elements in the array.
    ///
    /// Returns `None` if the array is empty.
    ///
    /// **Panics** if `q` is not between `0.` and `1.` (inclusive).
    fn quantile_mut<I>(&mut self, q: f64) -> Option<A>
    where
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>;
}

impl<A, S> Quantile1dExt<A, S> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
{
    fn quantile_mut<I>(&mut self, q: f64) -> Option<A>
    where
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>,
    {
        if self.is_empty() {
            None
        } else {
            Some(self.quantile_axis_mut::<I>(Axis(0), q).into_scalar())
        }
    }
}
