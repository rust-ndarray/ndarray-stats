use self::interpolate::{higher_index, lower_index, Interpolate};
use super::sort::get_many_from_sorted_mut_unchecked;
use crate::errors::QuantileError;
use crate::errors::{EmptyInput, MinMaxError, MinMaxError::UndefinedOrder};
use crate::{MaybeNan, MaybeNanExt};
use ndarray::prelude::*;
use ndarray::{Data, DataMut, RemoveAxis, Zip};
use noisy_float::types::N64;
use std::cmp;

/// Quantile methods for `ArrayBase`.
pub trait QuantileExt<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    /// Finds the index of the minimum value of the array.
    ///
    /// Returns `Err(MinMaxError::UndefinedOrder)` if any of the pairwise
    /// orderings tested by the function are undefined. (For example, this
    /// occurs if there are any floating-point NaN values in the array.)
    ///
    /// Returns `Err(MinMaxError::EmptyInput)` if the array is empty.
    ///
    /// Even if there are multiple (equal) elements that are minima, only one
    /// index is returned. (Which one is returned is unspecified and may depend
    /// on the memory layout of the array.)
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::QuantileExt;
    ///
    /// let a = array![[1., 3., 5.],
    ///                [2., 0., 6.]];
    /// assert_eq!(a.argmin(), Ok((1, 1)));
    /// ```
    fn argmin(&self) -> Result<D::Pattern, MinMaxError>
    where
        A: PartialOrd;

    /// Finds the index of the minimum value of the array skipping NaN values.
    ///
    /// Returns `Err(EmptyInput)` if the array is empty or none of the values in the array
    /// are non-NaN values.
    ///
    /// Even if there are multiple (equal) elements that are minima, only one
    /// index is returned. (Which one is returned is unspecified and may depend
    /// on the memory layout of the array.)
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::QuantileExt;
    ///
    /// let a = array![[::std::f64::NAN, 3., 5.],
    ///                [2., 0., 6.]];
    /// assert_eq!(a.argmin_skipnan(), Ok((1, 1)));
    /// ```
    fn argmin_skipnan(&self) -> Result<D::Pattern, EmptyInput>
    where
        A: MaybeNan,
        A::NotNan: Ord;

    /// Finds the elementwise minimum of the array.
    ///
    /// Returns `Err(MinMaxError::UndefinedOrder)` if any of the pairwise
    /// orderings tested by the function are undefined. (For example, this
    /// occurs if there are any floating-point NaN values in the array.)
    ///
    /// Returns `Err(MinMaxError::EmptyInput)` if the array is empty.
    ///
    /// Even if there are multiple (equal) elements that are minima, only one
    /// is returned. (Which one is returned is unspecified and may depend on
    /// the memory layout of the array.)
    fn min(&self) -> Result<&A, MinMaxError>
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
    /// Returns `Err(MinMaxError::UndefinedOrder)` if any of the pairwise
    /// orderings tested by the function are undefined. (For example, this
    /// occurs if there are any floating-point NaN values in the array.)
    ///
    /// Returns `Err(MinMaxError::EmptyInput)` if the array is empty.
    ///
    /// Even if there are multiple (equal) elements that are maxima, only one
    /// index is returned. (Which one is returned is unspecified and may depend
    /// on the memory layout of the array.)
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::QuantileExt;
    ///
    /// let a = array![[1., 3., 7.],
    ///                [2., 5., 6.]];
    /// assert_eq!(a.argmax(), Ok((0, 2)));
    /// ```
    fn argmax(&self) -> Result<D::Pattern, MinMaxError>
    where
        A: PartialOrd;

    /// Finds the index of the maximum value of the array skipping NaN values.
    ///
    /// Returns `Err(EmptyInput)` if the array is empty or none of the values in the array
    /// are non-NaN values.
    ///
    /// Even if there are multiple (equal) elements that are maxima, only one
    /// index is returned. (Which one is returned is unspecified and may depend
    /// on the memory layout of the array.)
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::QuantileExt;
    ///
    /// let a = array![[::std::f64::NAN, 3., 5.],
    ///                [2., 0., 6.]];
    /// assert_eq!(a.argmax_skipnan(), Ok((1, 2)));
    /// ```
    fn argmax_skipnan(&self) -> Result<D::Pattern, EmptyInput>
    where
        A: MaybeNan,
        A::NotNan: Ord;

    /// Finds the elementwise maximum of the array.
    ///
    /// Returns `Err(MinMaxError::UndefinedOrder)` if any of the pairwise
    /// orderings tested by the function are undefined. (For example, this
    /// occurs if there are any floating-point NaN values in the array.)
    ///
    /// Returns `Err(EmptyInput)` if the array is empty.
    ///
    /// Even if there are multiple (equal) elements that are maxima, only one
    /// is returned. (Which one is returned is unspecified and may depend on
    /// the memory layout of the array.)
    fn max(&self) -> Result<&A, MinMaxError>
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
    /// value depending on the `interpolate` strategy.
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
    /// Returns `Err(EmptyInput)` when the specified axis has length 0.
    ///
    /// Returns `Err(InvalidQuantile(q))` if `q` is not between `0.` and `1.` (inclusive).
    ///
    /// **Panics** if `axis` is out of bounds.
    fn quantile_axis_mut<I>(
        &mut self,
        axis: Axis,
        q: N64,
        interpolate: &I,
    ) -> Result<Array<A, D::Smaller>, QuantileError>
    where
        D: RemoveAxis,
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>;

    /// A bulk version of [`quantile_axis_mut`], optimized to retrieve multiple
    /// quantiles at once.
    ///
    /// Returns an `Array`, where subviews along `axis` of the array correspond
    /// to the elements of `qs`.
    ///
    /// See [`quantile_axis_mut`] for additional details on quantiles and the algorithm
    /// used to retrieve them.
    ///
    /// Returns `Err(EmptyInput)` when the specified axis has length 0.
    ///
    /// Returns `Err(InvalidQuantile(q))` if any `q` in `qs` is not between `0.` and `1.` (inclusive).
    ///
    /// **Panics** if `axis` is out of bounds.
    ///
    /// [`quantile_axis_mut`]: #tymethod.quantile_axis_mut
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::{array, aview1, Axis};
    /// use ndarray_stats::{QuantileExt, interpolate::Nearest};
    /// use noisy_float::types::n64;
    ///
    /// let mut data = array![[3, 4, 5], [6, 7, 8]];
    /// let axis = Axis(1);
    /// let qs = &[n64(0.3), n64(0.7)];
    /// let quantiles = data.quantiles_axis_mut(axis, &aview1(qs), &Nearest).unwrap();
    /// for (&q, quantile) in qs.iter().zip(quantiles.axis_iter(axis)) {
    ///     assert_eq!(quantile, data.quantile_axis_mut(axis, q, &Nearest).unwrap());
    /// }
    /// ```
    fn quantiles_axis_mut<S2, I>(
        &mut self,
        axis: Axis,
        qs: &ArrayBase<S2, Ix1>,
        interpolate: &I,
    ) -> Result<Array<A, D>, QuantileError>
    where
        D: RemoveAxis,
        A: Ord + Clone,
        S: DataMut,
        S2: Data<Elem = N64>,
        I: Interpolate<A>;

    /// Return the `q`th quantile of the data along the specified axis, skipping NaN values.
    ///
    /// See [`quantile_axis_mut`](#tymethod.quantile_axis_mut) for details.
    fn quantile_axis_skipnan_mut<I>(
        &mut self,
        axis: Axis,
        q: N64,
        interpolate: &I,
    ) -> Result<Array<A, D::Smaller>, QuantileError>
    where
        D: RemoveAxis,
        A: MaybeNan,
        A::NotNan: Clone + Ord,
        S: DataMut,
        I: Interpolate<A::NotNan>;

    private_decl! {}
}

impl<A, S, D> QuantileExt<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn argmin(&self) -> Result<D::Pattern, MinMaxError>
    where
        A: PartialOrd,
    {
        let mut current_min = self.first().ok_or(EmptyInput)?;
        let mut current_pattern_min = D::zeros(self.ndim()).into_pattern();

        for (pattern, elem) in self.indexed_iter() {
            if elem.partial_cmp(current_min).ok_or(UndefinedOrder)? == cmp::Ordering::Less {
                current_pattern_min = pattern;
                current_min = elem
            }
        }

        Ok(current_pattern_min)
    }

    fn argmin_skipnan(&self) -> Result<D::Pattern, EmptyInput>
    where
        A: MaybeNan,
        A::NotNan: Ord,
    {
        let mut pattern_min = D::zeros(self.ndim()).into_pattern();
        let min = self.indexed_fold_skipnan(None, |current_min, (pattern, elem)| {
            Some(match current_min {
                Some(m) if (m <= elem) => m,
                _ => {
                    pattern_min = pattern;
                    elem
                }
            })
        });
        if min.is_some() {
            Ok(pattern_min)
        } else {
            Err(EmptyInput)
        }
    }

    fn min(&self) -> Result<&A, MinMaxError>
    where
        A: PartialOrd,
    {
        let first = self.first().ok_or(EmptyInput)?;
        self.fold(Ok(first), |acc, elem| {
            let acc = acc?;
            match elem.partial_cmp(acc).ok_or(UndefinedOrder)? {
                cmp::Ordering::Less => Ok(elem),
                _ => Ok(acc),
            }
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

    fn argmax(&self) -> Result<D::Pattern, MinMaxError>
    where
        A: PartialOrd,
    {
        let mut current_max = self.first().ok_or(EmptyInput)?;
        let mut current_pattern_max = D::zeros(self.ndim()).into_pattern();

        for (pattern, elem) in self.indexed_iter() {
            if elem.partial_cmp(current_max).ok_or(UndefinedOrder)? == cmp::Ordering::Greater {
                current_pattern_max = pattern;
                current_max = elem
            }
        }

        Ok(current_pattern_max)
    }

    fn argmax_skipnan(&self) -> Result<D::Pattern, EmptyInput>
    where
        A: MaybeNan,
        A::NotNan: Ord,
    {
        let mut pattern_max = D::zeros(self.ndim()).into_pattern();
        let max = self.indexed_fold_skipnan(None, |current_max, (pattern, elem)| {
            Some(match current_max {
                Some(m) if m >= elem => m,
                _ => {
                    pattern_max = pattern;
                    elem
                }
            })
        });
        if max.is_some() {
            Ok(pattern_max)
        } else {
            Err(EmptyInput)
        }
    }

    fn max(&self) -> Result<&A, MinMaxError>
    where
        A: PartialOrd,
    {
        let first = self.first().ok_or(EmptyInput)?;
        self.fold(Ok(first), |acc, elem| {
            let acc = acc?;
            match elem.partial_cmp(acc).ok_or(UndefinedOrder)? {
                cmp::Ordering::Greater => Ok(elem),
                _ => Ok(acc),
            }
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

    fn quantiles_axis_mut<S2, I>(
        &mut self,
        axis: Axis,
        qs: &ArrayBase<S2, Ix1>,
        interpolate: &I,
    ) -> Result<Array<A, D>, QuantileError>
    where
        D: RemoveAxis,
        A: Ord + Clone,
        S: DataMut,
        S2: Data<Elem = N64>,
        I: Interpolate<A>,
    {
        // Minimize number of type parameters to avoid monomorphization bloat.
        fn quantiles_axis_mut<A, D, I>(
            mut data: ArrayViewMut<'_, A, D>,
            axis: Axis,
            qs: ArrayView1<'_, N64>,
            _interpolate: &I,
        ) -> Result<Array<A, D>, QuantileError>
        where
            D: RemoveAxis,
            A: Ord + Clone,
            I: Interpolate<A>,
        {
            for &q in qs {
                if !((q >= 0.) && (q <= 1.)) {
                    return Err(QuantileError::InvalidQuantile(q));
                }
            }

            let axis_len = data.len_of(axis);
            if axis_len == 0 {
                return Err(QuantileError::EmptyInput);
            }

            let mut results_shape = data.raw_dim();
            results_shape[axis.index()] = qs.len();
            if results_shape.size() == 0 {
                return Ok(Array::from_shape_vec(results_shape, Vec::new()).unwrap());
            }

            let mut searched_indexes = Vec::with_capacity(2 * qs.len());
            for &q in &qs {
                if I::needs_lower(q, axis_len) {
                    searched_indexes.push(lower_index(q, axis_len));
                }
                if I::needs_higher(q, axis_len) {
                    searched_indexes.push(higher_index(q, axis_len));
                }
            }
            searched_indexes.sort();
            searched_indexes.dedup();

            let mut results = Array::from_elem(results_shape, data.first().unwrap().clone());
            Zip::from(results.lanes_mut(axis))
                .and(data.lanes_mut(axis))
                .for_each(|mut results, mut data| {
                    let index_map =
                        get_many_from_sorted_mut_unchecked(&mut data, &searched_indexes);
                    for (result, &q) in results.iter_mut().zip(qs) {
                        let lower = if I::needs_lower(q, axis_len) {
                            Some(index_map[&lower_index(q, axis_len)].clone())
                        } else {
                            None
                        };
                        let higher = if I::needs_higher(q, axis_len) {
                            Some(index_map[&higher_index(q, axis_len)].clone())
                        } else {
                            None
                        };
                        *result = I::interpolate(lower, higher, q, axis_len);
                    }
                });
            Ok(results)
        }

        quantiles_axis_mut(self.view_mut(), axis, qs.view(), interpolate)
    }

    fn quantile_axis_mut<I>(
        &mut self,
        axis: Axis,
        q: N64,
        interpolate: &I,
    ) -> Result<Array<A, D::Smaller>, QuantileError>
    where
        D: RemoveAxis,
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>,
    {
        self.quantiles_axis_mut(axis, &aview1(&[q]), interpolate)
            .map(|a| a.index_axis_move(axis, 0))
    }

    fn quantile_axis_skipnan_mut<I>(
        &mut self,
        axis: Axis,
        q: N64,
        interpolate: &I,
    ) -> Result<Array<A, D::Smaller>, QuantileError>
    where
        D: RemoveAxis,
        A: MaybeNan,
        A::NotNan: Clone + Ord,
        S: DataMut,
        I: Interpolate<A::NotNan>,
    {
        if !((q >= 0.) && (q <= 1.)) {
            return Err(QuantileError::InvalidQuantile(q));
        }

        if self.len_of(axis) == 0 {
            return Err(QuantileError::EmptyInput);
        }

        let quantile = self.map_axis_mut(axis, |lane| {
            let mut not_nan = A::remove_nan_mut(lane);
            A::from_not_nan_opt(if not_nan.is_empty() {
                None
            } else {
                Some(
                    not_nan
                        .quantile_axis_mut::<I>(Axis(0), q, interpolate)
                        .unwrap()
                        .into_scalar(),
                )
            })
        });
        Ok(quantile)
    }

    private_impl! {}
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
    /// value depending on the `interpolate` strategy.
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
    /// Returns `Err(EmptyInput)` if the array is empty.
    ///
    /// Returns `Err(InvalidQuantile(q))` if `q` is not between `0.` and `1.` (inclusive).
    fn quantile_mut<I>(&mut self, q: N64, interpolate: &I) -> Result<A, QuantileError>
    where
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>;

    /// A bulk version of [`quantile_mut`], optimized to retrieve multiple
    /// quantiles at once.
    ///
    /// Returns an `Array`, where the elements of the array correspond to the
    /// elements of `qs`.
    ///
    /// Returns `Err(EmptyInput)` if the array is empty.
    ///
    /// Returns `Err(InvalidQuantile(q))` if any `q` in
    /// `qs` is not between `0.` and `1.` (inclusive).
    ///
    /// See [`quantile_mut`] for additional details on quantiles and the algorithm
    /// used to retrieve them.
    ///
    /// [`quantile_mut`]: #tymethod.quantile_mut
    fn quantiles_mut<S2, I>(
        &mut self,
        qs: &ArrayBase<S2, Ix1>,
        interpolate: &I,
    ) -> Result<Array1<A>, QuantileError>
    where
        A: Ord + Clone,
        S: DataMut,
        S2: Data<Elem = N64>,
        I: Interpolate<A>;

    private_decl! {}
}

impl<A, S> Quantile1dExt<A, S> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
{
    fn quantile_mut<I>(&mut self, q: N64, interpolate: &I) -> Result<A, QuantileError>
    where
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>,
    {
        Ok(self
            .quantile_axis_mut(Axis(0), q, interpolate)?
            .into_scalar())
    }

    fn quantiles_mut<S2, I>(
        &mut self,
        qs: &ArrayBase<S2, Ix1>,
        interpolate: &I,
    ) -> Result<Array1<A>, QuantileError>
    where
        A: Ord + Clone,
        S: DataMut,
        S2: Data<Elem = N64>,
        I: Interpolate<A>,
    {
        self.quantiles_axis_mut(Axis(0), qs, interpolate)
    }

    private_impl! {}
}

pub mod interpolate;
