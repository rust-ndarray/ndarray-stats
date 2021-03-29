use indexmap::IndexMap;
use ndarray::prelude::*;
use ndarray::{Data, DataMut, Slice};
use rand::prelude::*;
use rand::thread_rng;

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

    /// Partitions the array in increasing order based on the value initially
    /// located at `pivot_index` and returns the new index of the value.
    ///
    /// The elements are rearranged in such a way that the value initially
    /// located at `pivot_index` is moved to the position it would be in an
    /// array sorted in increasing order. The return value is the new index of
    /// the value after rearrangement. All elements smaller than the value are
    /// moved to its left and all elements equal or greater than the value are
    /// moved to its right. The ordering of the elements in the two partitions
    /// is undefined.
    ///
    /// `self` is shuffled **in place** to operate the desired partition:
    /// no copy of the array is allocated.
    ///
    /// The method uses Hoare's partition algorithm.
    /// Complexity: O(`n`), where `n` is the number of elements in the array.
    /// Average number of element swaps: n/6 - 1/3 (see
    /// [link](https://cs.stackexchange.com/questions/11458/quicksort-partitioning-hoare-vs-lomuto/11550))
    ///
    /// **Panics** if `pivot_index` is greater than or equal to `n`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::Sort1dExt;
    ///
    /// let mut data = array![3, 1, 4, 5, 2];
    /// let pivot_index = 2;
    /// let pivot_value = data[pivot_index];
    ///
    /// // Partition by the value located at `pivot_index`.
    /// let new_index = data.partition_mut(pivot_index);
    /// // The pivot value is now located at `new_index`.
    /// assert_eq!(data[new_index], pivot_value);
    /// // Elements less than that value are moved to the left.
    /// for i in 0..new_index {
    ///     assert!(data[i] < pivot_value);
    /// }
    /// // Elements greater than or equal to that value are moved to the right.
    /// for i in (new_index + 1)..data.len() {
    ///      assert!(data[i] >= pivot_value);
    /// }
    /// ```
    fn partition_mut(&mut self, pivot_index: usize) -> usize
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
            let mut rng = thread_rng();
            let pivot_index = rng.gen_range(0..n);
            let partition_index = self.partition_mut(pivot_index);
            if i < partition_index {
                self.slice_axis_mut(Axis(0), Slice::from(..partition_index))
                    .get_from_sorted_mut(i)
            } else if i == partition_index {
                self[i].clone()
            } else {
                self.slice_axis_mut(Axis(0), Slice::from(partition_index + 1..))
                    .get_from_sorted_mut(i - (partition_index + 1))
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

    fn partition_mut(&mut self, pivot_index: usize) -> usize
    where
        A: Ord + Clone,
        S: DataMut,
    {
        let pivot_value = self[pivot_index].clone();
        self.swap(pivot_index, 0);
        let n = self.len();
        let mut i = 1;
        let mut j = n - 1;
        loop {
            loop {
                if i > j {
                    break;
                }
                if self[i] >= pivot_value {
                    break;
                }
                i += 1;
            }
            while pivot_value <= self[j] {
                if j == 1 {
                    break;
                }
                j -= 1;
            }
            if i >= j {
                break;
            } else {
                self.swap(i, j);
                i += 1;
                j -= 1;
            }
        }
        self.swap(0, i - 1);
        i - 1
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

    // We pick a random pivot index: the corresponding element is the pivot value
    let mut rng = thread_rng();
    let pivot_index = rng.gen_range(0..n);

    // We partition the array with respect to the pivot value.
    // The pivot value moves to `array_partition_index`.
    // Elements strictly smaller than the pivot value have indexes < `array_partition_index`.
    // Elements greater or equal to the pivot value have indexes > `array_partition_index`.
    let array_partition_index = array.partition_mut(pivot_index);

    // We use a divide-and-conquer strategy, splitting the indexes we are
    // searching for (`indexes`) and the corresponding portions of the output
    // slice (`values`) into pieces with respect to `array_partition_index`.
    let (found_exact, index_split) = match indexes.binary_search(&array_partition_index) {
        Ok(index) => (true, index),
        Err(index) => (false, index),
    };
    let (smaller_indexes, other_indexes) = indexes.split_at_mut(index_split);
    let (smaller_values, other_values) = values.split_at_mut(index_split);
    let (bigger_indexes, bigger_values) = if found_exact {
        other_values[0] = array[array_partition_index].clone(); // Write exactly found value.
        (&mut other_indexes[1..], &mut other_values[1..])
    } else {
        (other_indexes, other_values)
    };

    // We search recursively for the values corresponding to strictly smaller
    // indexes to the left of `partition_index`.
    _get_many_from_sorted_mut_unchecked(
        array.slice_axis_mut(Axis(0), Slice::from(..array_partition_index)),
        smaller_indexes,
        smaller_values,
    );

    // We search recursively for the values corresponding to strictly bigger
    // indexes to the right of `partition_index`. Since only the right portion
    // of the array is passed in, the indexes need to be shifted by length of
    // the removed portion.
    bigger_indexes
        .iter_mut()
        .for_each(|x| *x -= array_partition_index + 1);
    _get_many_from_sorted_mut_unchecked(
        array.slice_axis_mut(Axis(0), Slice::from(array_partition_index + 1..)),
        bigger_indexes,
        bigger_values,
    );
}
