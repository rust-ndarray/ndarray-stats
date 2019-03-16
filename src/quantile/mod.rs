use self::interpolate::{higher_index, lower_index, Interpolate};
use super::sort::get_many_from_sorted_mut_unchecked;
use indexmap::{IndexMap, IndexSet};
use ndarray::prelude::*;
use ndarray::{Data, DataMut, RemoveAxis};
use noisy_float::types::N64;
use std::cmp;
use {MaybeNan, MaybeNanExt};

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
    /// Returns `None` when the specified axis has length 0.
    ///
    /// **Panics** if `axis` is out of bounds or if
    /// `q` is not between `0.` and `1.` (inclusive).
    fn quantile_axis_mut<I>(&mut self, axis: Axis, q: N64) -> Option<Array<A, D::Smaller>>
    where
        D: RemoveAxis,
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>;

    /// A bulk version of [quantile_axis_mut], optimized to retrieve multiple
    /// quantiles at once.
    /// It returns an IndexMap, with (quantile index, quantile over axis) as
    /// key-value pairs.
    ///
    /// The IndexMap is sorted with respect to quantile indexes in increasing order:
    /// this ordering is preserved when you iterate over it (using `iter`/`into_iter`).
    ///
    /// See [quantile_axis_mut] for additional details on quantiles and the algorithm
    /// used to retrieve them.
    ///
    /// Returns `None` when the specified axis has length 0.
    ///
    /// **Panics** if `axis` is out of bounds or if
    /// any `q` in `qs` is not between `0.` and `1.` (inclusive).
    ///
    /// [quantile_axis_mut]: ##tymethod.quantile_axis_mut
    fn quantiles_axis_mut<I>(
        &mut self,
        axis: Axis,
        qs: &[N64],
    ) -> Option<IndexMap<N64, Array<A, D::Smaller>>>
    where
        D: RemoveAxis,
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>;

    /// Return the `q`th quantile of the data along the specified axis, skipping NaN values.
    ///
    /// See [`quantile_axis_mut`](##tymethod.quantile_axis_mut) for details.
    fn quantile_axis_skipnan_mut<I>(&mut self, axis: Axis, q: N64) -> Option<Array<A, D::Smaller>>
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

    fn quantiles_axis_mut<I>(
        &mut self,
        axis: Axis,
        qs: &[N64],
    ) -> Option<IndexMap<N64, Array<A, D::Smaller>>>
    where
        D: RemoveAxis,
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>,
    {
        assert!(qs.iter().all(|x| (*x >= 0.) && (*x <= 1.)));

        let axis_len = self.len_of(axis);
        if axis_len == 0 {
            return None;
        }

        let mut deduped_qs: Vec<N64> = qs.to_vec();
        deduped_qs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        deduped_qs.dedup();

        // IndexSet preserves insertion order:
        // - indexes will stay sorted;
        // - we avoid index duplication.
        let mut searched_indexes = IndexSet::new();
        for q in deduped_qs.iter() {
            if I::needs_lower(*q, axis_len) {
                searched_indexes.insert(lower_index(*q, axis_len));
            }
            if I::needs_higher(*q, axis_len) {
                searched_indexes.insert(higher_index(*q, axis_len));
            }
        }
        let searched_indexes: Vec<usize> = searched_indexes.into_iter().collect();

        // Retrieve the values corresponding to each index for each slice along the specified axis
        // For each 1-dimensional slice along the specified axis we get back an IndexMap
        // which can be used to retrieve the desired values using searched_indexes
        let values = self.map_axis_mut(axis, |mut x| {
            get_many_from_sorted_mut_unchecked(&mut x, &searched_indexes)
        });

        // Combine the retrieved values according to specified interpolation strategy to
        // get the desired quantiles
        let mut results = IndexMap::new();
        for q in qs {
            let lower = if I::needs_lower(*q, axis_len) {
                Some(values.map(|x| x[&lower_index(*q, axis_len)].clone()))
            } else {
                None
            };
            let higher = if I::needs_higher(*q, axis_len) {
                Some(values.map(|x| x[&higher_index(*q, axis_len)].clone()))
            } else {
                None
            };
            let interpolated = I::interpolate(lower, higher, *q, axis_len);
            results.insert(*q, interpolated);
        }
        Some(results)
    }

    fn quantile_axis_mut<I>(&mut self, axis: Axis, q: N64) -> Option<Array<A, D::Smaller>>
    where
        D: RemoveAxis,
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>,
    {
        self.quantiles_axis_mut::<I>(axis, &[q])
            .map(|x| x.into_iter().next().unwrap().1)
    }

    fn quantile_axis_skipnan_mut<I>(&mut self, axis: Axis, q: N64) -> Option<Array<A, D::Smaller>>
    where
        D: RemoveAxis,
        A: MaybeNan,
        A::NotNan: Clone + Ord,
        S: DataMut,
        I: Interpolate<A::NotNan>,
    {
        if self.len_of(axis) == 0 {
            return None;
        }
        let quantile = self.map_axis_mut(axis, |lane| {
            let mut not_nan = A::remove_nan_mut(lane);
            A::from_not_nan_opt(if not_nan.is_empty() {
                None
            } else {
                Some(
                    not_nan
                        .quantile_axis_mut::<I>(Axis(0), q)
                        .unwrap()
                        .into_scalar(),
                )
            })
        });
        Some(quantile)
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
    fn quantile_mut<I>(&mut self, q: N64) -> Option<A>
    where
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>;

    /// A bulk version of [quantile_mut], optimized to retrieve multiple
    /// quantiles at once.
    /// It returns an IndexMap, with (quantile index, quantile value) as
    /// key-value pairs.
    ///
    /// The IndexMap is sorted with respect to quantile indexes in increasing order:
    /// this ordering is preserved when you iterate over it (using `iter`/`into_iter`).
    ///
    /// It returns `None` if the array is empty.
    ///
    /// See [quantile_mut] for additional details on quantiles and the algorithm
    /// used to retrieve them.
    ///
    /// **Panics** if any `q` in `qs` is not between `0.` and `1.` (inclusive).
    ///
    /// [quantile_mut]: ##tymethod.quantile_mut
    fn quantiles_mut<I>(&mut self, qs: &[N64]) -> Option<IndexMap<N64, A>>
    where
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>;
}

impl<A, S> Quantile1dExt<A, S> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
{
    fn quantile_mut<I>(&mut self, q: N64) -> Option<A>
    where
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>,
    {
        self.quantile_axis_mut::<I>(Axis(0), q)
            .map(|v| v.into_scalar())
    }

    fn quantiles_mut<I>(&mut self, qs: &[N64]) -> Option<IndexMap<N64, A>>
    where
        A: Ord + Clone,
        S: DataMut,
        I: Interpolate<A>,
    {
        self.quantiles_axis_mut::<I>(Axis(0), qs)
            .map(|v| v.into_iter().map(|x| (x.0, x.1.into_scalar())).collect())
    }
}

pub mod interpolate;
