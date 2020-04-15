#![warn(missing_docs, clippy::all, clippy::pedantic)]

use ndarray::prelude::*;
use std::ops::{Index, Range};

/// A sorted collection of type `A` elements used to represent the boundaries of intervals, i.e.
/// [`Bins`] on a 1-dimensional axis.
///
/// **Note** that all intervals are left-closed and right-open. See examples below.
///
/// # Examples
///
/// ```
/// use ndarray_stats::histogram::{Bins, Edges};
/// use noisy_float::types::n64;
///
/// let unit_edges = Edges::from(vec![n64(0.), n64(1.)]);
/// let unit_interval = Bins::new(unit_edges);
/// // left-closed
/// assert_eq!(
///     unit_interval.range_of(&n64(0.)).unwrap(),
///     n64(0.)..n64(1.),
/// );
/// // right-open
/// assert_eq!(
///     unit_interval.range_of(&n64(1.)),
///     None
/// );
/// ```
///
/// [`Bins`]: struct.Bins.html
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Edges<A: Ord> {
    edges: Vec<A>,
}

impl<A: Ord> From<Vec<A>> for Edges<A> {
    /// Converts a `Vec<A>` into an `Edges<A>`, consuming the edges.
    /// The vector will be sorted in increasing order using an unstable sorting algorithm, with
    /// duplicates removed.
    ///
    /// # Current implementation
    ///
    /// The current sorting algorithm is the same as [`std::slice::sort_unstable()`][sort],
    /// which is based on [pattern-defeating quicksort][pdqsort].
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place (i.e., does not allocate)
    /// , and O(n log n) worst-case.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::Edges;
    ///
    /// let edges = Edges::from(array![1, 15, 10, 10, 20]);
    /// // The array gets sorted!
    /// assert_eq!(
    ///     edges[2],
    ///     15
    /// );
    /// ```
    ///
    /// [sort]: https://doc.rust-lang.org/stable/std/primitive.slice.html#method.sort_unstable
    /// [pdqsort]: https://github.com/orlp/pdqsort
    fn from(mut edges: Vec<A>) -> Self {
        // sort the array in-place
        edges.sort_unstable();
        // remove duplicates
        edges.dedup();
        Edges { edges }
    }
}

impl<A: Ord + Clone> From<Array1<A>> for Edges<A> {
    /// Converts an `Array1<A>` into an `Edges<A>`, consuming the 1-dimensional array.
    /// The array will be sorted in increasing order using an unstable sorting algorithm, with
    /// duplicates removed.
    ///
    /// # Current implementation
    ///
    /// The current sorting algorithm is the same as [`std::slice::sort_unstable()`][sort],
    /// which is based on [pattern-defeating quicksort][pdqsort].
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place (i.e., does not allocate)
    /// , and O(n log n) worst-case.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::Edges;
    ///
    /// let edges = Edges::from(vec![1, 15, 10, 20]);
    /// // The vec gets sorted!
    /// assert_eq!(
    ///     edges[1],
    ///     10
    /// );
    /// ```
    ///
    /// [sort]: https://doc.rust-lang.org/stable/std/primitive.slice.html#method.sort_unstable
    /// [pdqsort]: https://github.com/orlp/pdqsort
    fn from(edges: Array1<A>) -> Self {
        let edges = edges.to_vec();
        Self::from(edges)
    }
}

impl<A: Ord> Index<usize> for Edges<A> {
    type Output = A;

    /// Returns a reference to the `i`-th edge in `self`.
    ///
    /// # Panics
    ///
    /// Panics if the index `i` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::Edges;
    ///
    /// let edges = Edges::from(vec![1, 5, 10, 20]);
    /// assert_eq!(
    ///     edges[1],
    ///     5
    /// );
    /// ```
    fn index(&self, i: usize) -> &Self::Output {
        &self.edges[i]
    }
}

impl<A: Ord> Edges<A> {
    /// Returns the number of edges in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::Edges;
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(0.), n64(1.), n64(3.)]);
    /// assert_eq!(
    ///     edges.len(),
    ///     3
    /// );
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Returns `true` if `self` contains no edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::Edges;
    /// use noisy_float::types::{N64, n64};
    ///
    /// let edges = Edges::<N64>::from(vec![]);
    /// assert_eq!(edges.is_empty(), true);
    ///
    /// let edges = Edges::from(vec![n64(0.), n64(2.), n64(5.)]);
    /// assert_eq!(edges.is_empty(), false);
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Returns an immutable 1-dimensional array view of edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::Edges;
    ///
    /// let edges = Edges::from(vec![0, 5, 3]);
    /// assert_eq!(
    ///     edges.as_array_view(),
    ///     array![0, 3, 5].view()
    /// );
    /// ```
    #[must_use]
    pub fn as_array_view(&self) -> ArrayView1<'_, A> {
        ArrayView1::from(&self.edges)
    }

    /// Returns indices of two consecutive `edges` in `self`, if the interval they represent
    /// contains the given `value`, or returns `None` otherwise.
    ///
    /// That is to say, it returns
    /// - `Some((left, right))`, where `left` and `right` are the indices of two consecutive edges
    /// in `self` and `right == left + 1`, if `self[left] <= value < self[right]`;
    /// - `None`, otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::Edges;
    ///
    /// let edges = Edges::from(vec![0, 2, 3]);
    /// // `1` is in the interval [0, 2), whose indices are (0, 1)
    /// assert_eq!(
    ///     edges.indices_of(&1),
    ///     Some((0, 1))
    /// );
    /// // `5` is not in any of intervals
    /// assert_eq!(
    ///     edges.indices_of(&5),
    ///     None
    /// );
    /// ```
    pub fn indices_of(&self, value: &A) -> Option<(usize, usize)> {
        // binary search for the correct bin
        let n_edges = self.len();
        match self.edges.binary_search(value) {
            Ok(i) if i == n_edges - 1 => None,
            Ok(i) => Some((i, i + 1)),
            Err(i) => match i {
                0 => None,
                j if j == n_edges => None,
                j => Some((j - 1, j)),
            },
        }
    }

    /// Returns an iterator over the `edges` in `self`.
    pub fn iter(&self) -> impl Iterator<Item = &A> {
        self.edges.iter()
    }
}

/// A sorted collection of non-overlapping 1-dimensional intervals.
///
/// **Note** that all intervals are left-closed and right-open.
///
/// # Examples
///
/// ```
/// use ndarray_stats::histogram::{Edges, Bins};
/// use noisy_float::types::n64;
///
/// let edges = Edges::from(vec![n64(0.), n64(1.), n64(2.)]);
/// let bins = Bins::new(edges);
/// // first bin
/// assert_eq!(
///     bins.index(0),
///     n64(0.)..n64(1.) // n64(1.) is not included in the bin!
/// );
/// // second bin
/// assert_eq!(
///     bins.index(1),
///     n64(1.)..n64(2.)
/// );
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Bins<A: Ord> {
    edges: Edges<A>,
}

impl<A: Ord> Bins<A> {
    /// Returns a `Bins` instance where each bin corresponds to two consecutive members of the given
    /// [`Edges`], consuming the edges.
    ///
    /// [`Edges`]: struct.Edges.html
    #[must_use]
    pub fn new(edges: Edges<A>) -> Self {
        Bins { edges }
    }

    /// Returns the number of bins in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::{Edges, Bins};
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(0.), n64(1.), n64(2.)]);
    /// let bins = Bins::new(edges);
    /// assert_eq!(
    ///     bins.len(),
    ///     2
    /// );
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        match self.edges.len() {
            0 => 0,
            n => n - 1,
        }
    }

    /// Returns `true` if the number of bins is zero, i.e. if the number of edges is 0 or 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::{Edges, Bins};
    /// use noisy_float::types::{N64, n64};
    ///
    /// // At least 2 edges is needed to represent 1 interval
    /// let edges = Edges::from(vec![n64(0.), n64(1.), n64(3.)]);
    /// let bins = Bins::new(edges);
    /// assert_eq!(bins.is_empty(), false);
    ///
    /// // No valid interval == Empty
    /// let edges = Edges::<N64>::from(vec![]);
    /// let bins = Bins::new(edges);
    /// assert_eq!(bins.is_empty(), true);
    /// let edges = Edges::from(vec![n64(0.)]);
    /// let bins = Bins::new(edges);
    /// assert_eq!(bins.is_empty(), true);
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the index of the bin in `self` that contains the given `value`,
    /// or returns `None` if `value` does not belong to any bins in `self`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use ndarray_stats::histogram::{Edges, Bins};
    ///
    /// let edges = Edges::from(vec![0, 2, 4, 6]);
    /// let bins = Bins::new(edges);
    /// let value = 1;
    /// // The first bin [0, 2) contains `1`
    /// assert_eq!(
    ///     bins.index_of(&1),
    ///     Some(0)
    /// );
    /// // No bin contains 100
    /// assert_eq!(
    ///     bins.index_of(&100),
    ///     None
    /// )
    /// ```
    ///
    /// Chaining [`Bins::index`] and [`Bins::index_of`] to get the boundaries of the bin containing
    /// the value:
    ///
    /// ```
    /// # use ndarray_stats::histogram::{Edges, Bins};
    /// # let edges = Edges::from(vec![0, 2, 4, 6]);
    /// # let bins = Bins::new(edges);
    /// # let value = 1;
    /// assert_eq!(
    ///     // using `Option::map` to avoid panic on index out-of-bounds
    ///     bins.index_of(&1).map(|i| bins.index(i)),
    ///     Some(0..2)
    /// );
    /// ```
    pub fn index_of(&self, value: &A) -> Option<usize> {
        self.edges.indices_of(value).map(|t| t.0)
    }

    /// Returns a range as the bin which contains the given `value`, or returns `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::{Edges, Bins};
    ///
    /// let edges = Edges::from(vec![0, 2, 4, 6]);
    /// let bins = Bins::new(edges);
    /// // [0, 2) contains `1`
    /// assert_eq!(
    ///     bins.range_of(&1),
    ///     Some(0..2)
    /// );
    /// // `10` is not in any interval
    /// assert_eq!(
    ///     bins.range_of(&10),
    ///     None
    /// );
    /// ```
    pub fn range_of(&self, value: &A) -> Option<Range<A>>
    where
        A: Clone,
    {
        let edges_indexes = self.edges.indices_of(value);
        edges_indexes.map(|(left, right)| Range {
            start: self.edges[left].clone(),
            end: self.edges[right].clone(),
        })
    }

    /// Returns a range as the bin at the given `index` position.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::{Edges, Bins};
    ///
    /// let edges = Edges::from(vec![1, 5, 10, 20]);
    /// let bins = Bins::new(edges);
    /// assert_eq!(
    ///     bins.index(1),
    ///     5..10
    /// );
    /// ```
    #[must_use]
    pub fn index(&self, index: usize) -> Range<A>
    where
        A: Clone,
    {
        // It was not possible to implement this functionality
        // using the `Index` trait unless we were willing to
        // allocate a `Vec<Range<A>>` in the struct.
        // Index, in fact, forces you to return a reference.
        Range {
            start: self.edges[index].clone(),
            end: self.edges[index + 1].clone(),
        }
    }
}

#[cfg(test)]
mod edges_tests {
    use super::{Array1, Edges};
    use quickcheck_macros::quickcheck;
    use std::collections::BTreeSet;
    use std::iter::FromIterator;

    #[quickcheck]
    fn check_sorted_from_vec(v: Vec<i32>) -> bool {
        let edges = Edges::from(v);
        let n = edges.len();
        for i in 1..n {
            if edges[i - 1] > edges[i] {
                return false;
            }
        }
        true
    }

    #[quickcheck]
    fn check_sorted_from_array(v: Vec<i32>) -> bool {
        let a = Array1::from(v);
        let edges = Edges::from(a);
        let n = edges.len();
        for i in 1..n {
            if edges[i - 1] > edges[i] {
                return false;
            }
        }
        true
    }

    #[quickcheck]
    fn edges_are_right_open(v: Vec<i32>) -> bool {
        let edges = Edges::from(v);
        let view = edges.as_array_view();
        if view.is_empty() {
            true
        } else {
            let last = view[view.len() - 1];
            edges.indices_of(&last).is_none()
        }
    }

    #[quickcheck]
    fn edges_are_left_closed(v: Vec<i32>) -> bool {
        let edges = Edges::from(v);
        if let 1 = edges.len() {
            true
        } else {
            let view = edges.as_array_view();
            if view.is_empty() {
                true
            } else {
                let first = view[0];
                edges.indices_of(&first).is_some()
            }
        }
    }

    #[quickcheck]
    #[allow(clippy::needless_pass_by_value)]
    fn edges_are_deduped(v: Vec<i32>) -> bool {
        let unique_elements = BTreeSet::from_iter(v.iter());
        let edges = Edges::from(v.clone());
        let view = edges.as_array_view();
        let unique_edges = BTreeSet::from_iter(view.iter());
        unique_edges == unique_elements
    }
}

#[cfg(test)]
mod bins_tests {
    use super::{Bins, Edges};

    #[test]
    #[should_panic]
    #[allow(unused_must_use)]
    fn get_panics_for_out_of_bounds_indexes() {
        let edges = Edges::from(vec![0]);
        let bins = Bins::new(edges);
        // we need at least two edges to make a valid bin!
        bins.index(0);
    }
}
