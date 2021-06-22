use indexmap::IndexMap;
use ndarray::prelude::*;
use ndarray::{Data, DataMut, Slice};

/// Methods for sorting and partitioning 1-D arrays.
pub trait Sort1dExt<A, S>
where
    S: Data<Elem = A>,
{
    /// Return the element that would occupy the `i`-th position if
    /// the array were sorted in increasing order.
    ///
    /// The array is shuffled **in place** to retrieve the desired element:
    /// no copy of the array is allocated.
    /// After the shuffling, all elements with an index smaller than `i`
    /// are smaller than the desired element, while all elements with
    /// an index greater or equal than `i` are greater than or equal
    /// to the desired element.
    ///
    /// No other assumptions should be made on the ordering of the
    /// elements after this computation.
    ///
    /// Complexity ([quickselect](https://en.wikipedia.org/wiki/Quickselect)):
    /// - average case: O(`n`);
    /// - worst case: O(`n`^2);
    /// where n is the number of elements in the array.
    ///
    /// **Panics** if `i` is greater than or equal to `n`.
    fn get_from_sorted_mut(&mut self, i: usize) -> A
    where
        A: Ord + Clone,
        S: DataMut;

    /// A bulk version of [`get_from_sorted_mut`], optimized to retrieve multiple
    /// indexes at once.
    /// It returns an `IndexMap`, with indexes as keys and retrieved elements as
    /// values.
    /// The `IndexMap` is sorted with respect to indexes in increasing order:
    /// this ordering is preserved when you iterate over it (using `iter`/`into_iter`).
    ///
    /// **Panics** if any element in `indexes` is greater than or equal to `n`,
    /// where `n` is the length of the array..
    ///
    /// [`get_from_sorted_mut`]: #tymethod.get_from_sorted_mut
    fn get_many_from_sorted_mut<S2>(&mut self, indexes: &ArrayBase<S2, Ix1>) -> IndexMap<usize, A>
    where
        A: Ord + Clone,
        S: DataMut,
        S2: Data<Elem = usize>;

    /// Partitions the array in increasing order at two skewed pivot values as 1st and 3rd element
    /// of a sorted sample of 5 equally spaced elements around the center and returns their indexes.
    /// For arrays of less than 42 elements the outermost elements serve as sample for pivot values.
    ///
    /// The elements are rearranged in such a way that the two pivot values are moved to the indexes
    /// they would be in an array sorted in increasing order. The return values are the new indexes
    /// of the values after rearrangement. All elements less than the values are moved to their left
    /// and all elements equal or greater than the values are moved to their right. The ordering of
    /// the elements in the three partitions is undefined.
    ///
    /// The array is shuffled **in place**, no copy of the array is allocated.
    ///
    /// This method performs [dual-pivot partitioning] with skewed pivot sampling.
    ///
    /// [dual-pivot partitioning]: https://www.wild-inter.net/publications/wild-2018b.pdf
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::Sort1dExt;
    ///
    /// let mut data = array![3, 1, 4, 5, 2];
    /// // Sorted pivot values.
    /// let (lower_value, upper_value) = (data[data.len() - 1], data[0]);
    ///
    /// // Partitions by the values located at `0` and `data.len() - 1`.
    /// let (lower_index, upper_index) = data.partition_mut();
    /// // The pivot values are now located at `lower_index` and `upper_index`.
    /// assert_eq!(data[lower_index], lower_value);
    /// assert_eq!(data[upper_index], upper_value);
    /// // Elements lower than the lower pivot value are moved to its left.
    /// for i in 0..lower_index {
    ///     assert!(data[i] < lower_value);
    /// }
    /// // Elements greater than or equal the lower pivot value and less than or equal the upper
    /// // pivot value are moved between the two pivot indexes.
    /// for i in lower_index + 1..upper_index {
    ///     assert!(lower_value <= data[i]);
    ///     assert!(data[i] <= upper_value);
    /// }
    /// // Elements greater than or equal the upper pivot value are moved to its right.
    /// for i in upper_index + 1..data.len() {
    ///     assert!(upper_value <= data[i]);
    /// }
    /// ```
    fn partition_mut(&mut self) -> (usize, usize)
    where
        A: Ord + Clone,
        S: DataMut;

    private_decl! {}
}

impl<A, S> Sort1dExt<A, S> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
{
    fn get_from_sorted_mut(&mut self, i: usize) -> A
    where
        A: Ord + Clone,
        S: DataMut,
    {
        let n = self.len();
        if n == 1 {
            self[0].clone()
        } else {
            let (lower_index, upper_index) = self.partition_mut();
            if i < lower_index {
                self.slice_axis_mut(Axis(0), Slice::from(..lower_index))
                    .get_from_sorted_mut(i)
            } else if i == lower_index {
                self[i].clone()
            } else if i < upper_index {
                self.slice_axis_mut(Axis(0), Slice::from(lower_index + 1..upper_index))
                    .get_from_sorted_mut(i - (lower_index + 1))
            } else if i == upper_index {
                self[i].clone()
            } else {
                self.slice_axis_mut(Axis(0), Slice::from(upper_index + 1..))
                    .get_from_sorted_mut(i - (upper_index + 1))
            }
        }
    }

    fn get_many_from_sorted_mut<S2>(&mut self, indexes: &ArrayBase<S2, Ix1>) -> IndexMap<usize, A>
    where
        A: Ord + Clone,
        S: DataMut,
        S2: Data<Elem = usize>,
    {
        let mut deduped_indexes: Vec<usize> = indexes.to_vec();
        deduped_indexes.sort_unstable();
        deduped_indexes.dedup();

        get_many_from_sorted_mut_unchecked(self, &deduped_indexes)
    }

    fn partition_mut(&mut self) -> (usize, usize)
    where
        A: Ord + Clone,
        S: DataMut,
    {
        let lowermost = 0;
        let uppermost = self.len() - 1;
        if self.len() < 42 {
            // Sort outermost elements and use them as pivots.
            if self[lowermost] > self[uppermost] {
                self.swap(lowermost, uppermost);
            }
        } else {
            // Sample indexes of 5 evenly spaced elements around the center element.
            let mut samples = [0; 5];
            // Assume array of at least 7 elements.
            let seventh = self.len() / (samples.len() + 2);
            samples[2] = self.len() / 2;
            samples[1] = samples[2] - seventh;
            samples[0] = samples[1] - seventh;
            samples[3] = samples[2] + seventh;
            samples[4] = samples[3] + seventh;
            // Use insertion sort for sample elements by looking up their indexes.
            for mut index in 1..samples.len() {
                while index > 0 && self[samples[index - 1]] > self[samples[index]] {
                    self.swap(samples[index - 1], samples[index]);
                    index -= 1;
                }
            }
            // Use 1st and 3rd element of sorted sample as skewed pivots.
            self.swap(lowermost, samples[0]);
            self.swap(uppermost, samples[2]);
        }
        // Increasing running and partition index starting after lower pivot.
        let mut index = lowermost + 1;
        let mut lower = lowermost + 1;
        // Decreasing partition index starting before upper pivot.
        let mut upper = uppermost - 1;
        // Swap elements at `index` into their partitions.
        while index <= upper {
            if self[index] < self[lowermost] {
                // Swap elements into lower partition.
                self.swap(index, lower);
                lower += 1;
            } else if self[index] >= self[uppermost] {
                // Search first element of upper partition.
                while self[upper] > self[uppermost] && index < upper {
                    upper -= 1;
                }
                // Swap elements into upper partition.
                self.swap(index, upper);
                if self[index] < self[lowermost] {
                    // Swap swapped elements into lower partition.
                    self.swap(index, lower);
                    lower += 1;
                }
                upper -= 1;
            }
            index += 1;
        }
        lower -= 1;
        upper += 1;
        // Swap pivots to their new indexes.
        self.swap(lowermost, lower);
        self.swap(uppermost, upper);
        // Lower and upper pivot index.
        (lower, upper)
    }

    private_impl! {}
}

/// To retrieve multiple indexes from the sorted array in an optimized fashion,
/// [get_many_from_sorted_mut] first of all sorts and deduplicates the
/// `indexes` vector.
///
/// `get_many_from_sorted_mut_unchecked` does not perform this sorting and
/// deduplication, assuming that the user has already taken care of it.
///
/// Useful when you have to call [get_many_from_sorted_mut] multiple times
/// using the same indexes.
///
/// [get_many_from_sorted_mut]: ../trait.Sort1dExt.html#tymethod.get_many_from_sorted_mut
pub(crate) fn get_many_from_sorted_mut_unchecked<A, S>(
    array: &mut ArrayBase<S, Ix1>,
    indexes: &[usize],
) -> IndexMap<usize, A>
where
    A: Ord + Clone,
    S: DataMut<Elem = A>,
{
    if indexes.is_empty() {
        return IndexMap::new();
    }

    // Since `!indexes.is_empty()` and indexes must be in-bounds, `array` must
    // be non-empty.
    let mut values = vec![array[0].clone(); indexes.len()];
    _get_many_from_sorted_mut_unchecked(array.view_mut(), &mut indexes.to_owned(), &mut values);

    // We convert the vector to a more search-friendly `IndexMap`.
    indexes.iter().cloned().zip(values.into_iter()).collect()
}

/// This is the recursive portion of `get_many_from_sorted_mut_unchecked`.
///
/// `indexes` is the list of indexes to get. `indexes` is mutable so that it
/// can be used as scratch space for this routine; the value of `indexes` after
/// calling this routine should be ignored.
///
/// `values` is a pre-allocated slice to use for writing the output. Its
/// initial element values are ignored.
fn _get_many_from_sorted_mut_unchecked<A>(
    mut array: ArrayViewMut1<'_, A>,
    indexes: &mut [usize],
    values: &mut [A],
) where
    A: Ord + Clone,
{
    let n = array.len();
    debug_assert!(n >= indexes.len()); // because indexes must be unique and in-bounds
    debug_assert_eq!(indexes.len(), values.len());

    if indexes.is_empty() {
        // Nothing to do in this case.
        return;
    }

    // At this point, `n >= 1` since `indexes.len() >= 1`.
    if n == 1 {
        // We can only reach this point if `indexes.len() == 1`, so we only
        // need to assign the single value, and then we're done.
        debug_assert_eq!(indexes.len(), 1);
        values[0] = array[0].clone();
        return;
    }

    // We partition the array with respect to the two pivot values. The pivot values move to
    // `lower_index` and `upper_index`.
    //
    // Elements strictly less than the lower pivot value have indexes < `lower_index`.
    //
    // Elements greater than or equal the lower pivot value and less than or equal the upper pivot
    // value have indexes > `lower_index` and < `upper_index`.
    //
    // Elements less than or equal the upper pivot value have indexes > `upper_index`.
    let (lower_index, upper_index) = array.partition_mut();

    // We use a divide-and-conquer strategy, splitting the indexes we are searching for (`indexes`)
    // and the corresponding portions of the output slice (`values`) into partitions with respect to
    // `lower_index` and `upper_index`.
    let (found_exact, split_index) = match indexes.binary_search(&lower_index) {
        Ok(index) => (true, index),
        Err(index) => (false, index),
    };
    let (lower_indexes, inner_indexes) = indexes.split_at_mut(split_index);
    let (lower_values, inner_values) = values.split_at_mut(split_index);
    let (upper_indexes, upper_values) = if found_exact {
        inner_values[0] = array[lower_index].clone(); // Write exactly found value.
        (&mut inner_indexes[1..], &mut inner_values[1..])
    } else {
        (inner_indexes, inner_values)
    };

    let (found_exact, split_index) = match upper_indexes.binary_search(&upper_index) {
        Ok(index) => (true, index),
        Err(index) => (false, index),
    };
    let (inner_indexes, upper_indexes) = upper_indexes.split_at_mut(split_index);
    let (inner_values, upper_values) = upper_values.split_at_mut(split_index);
    let (upper_indexes, upper_values) = if found_exact {
        upper_values[0] = array[upper_index].clone(); // Write exactly found value.
        (&mut upper_indexes[1..], &mut upper_values[1..])
    } else {
        (upper_indexes, upper_values)
    };

    // We search recursively for the values corresponding to indexes strictly less than
    // `lower_index` in the lower partition.
    _get_many_from_sorted_mut_unchecked(
        array.slice_axis_mut(Axis(0), Slice::from(..lower_index)),
        lower_indexes,
        lower_values,
    );

    // We search recursively for the values corresponding to indexes greater than or equal
    // `lower_index` in the inner partition, that is between the lower and upper partition. Since
    // only the inner partition of the array is passed in, the indexes need to be shifted by length
    // of the lower partition.
    inner_indexes.iter_mut().for_each(|x| *x -= lower_index + 1);
    _get_many_from_sorted_mut_unchecked(
        array.slice_axis_mut(Axis(0), Slice::from(lower_index + 1..upper_index)),
        inner_indexes,
        inner_values,
    );

    // We search recursively for the values corresponding to indexes greater than or equal
    // `upper_index` in the upper partition. Since only the upper partition of the array is passed
    // in, the indexes need to be shifted by the combined length of the lower and inner partition.
    upper_indexes.iter_mut().for_each(|x| *x -= upper_index + 1);
    _get_many_from_sorted_mut_unchecked(
        array.slice_axis_mut(Axis(0), Slice::from(upper_index + 1..)),
        upper_indexes,
        upper_values,
    );
}
