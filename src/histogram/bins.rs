use ndarray::prelude::*;
use std::ops::{Index, Range};

/// `Edges` is a sorted collection of `A` elements used
/// to represent the boundaries of intervals ([`Bins`]) on
/// a 1-dimensional axis.
///
/// [`Bins`]: struct.Bins.html
/// # Example:
///
/// ```
/// extern crate ndarray_stats;
/// extern crate noisy_float;
/// use ndarray_stats::histogram::{Edges, Bins};
/// use noisy_float::types::n64;
///
/// let unit_edges = Edges::from(vec![n64(0.), n64(1.)]);
/// let unit_interval = Bins::new(unit_edges);
/// // left inclusive
/// assert_eq!(
///     unit_interval.range(&n64(0.)).unwrap(),
///     n64(0.)..n64(1.),
/// );
/// // right exclusive
/// assert_eq!(
///     unit_interval.range(&n64(1.)),
///     None
/// );
/// ```
#[derive(Clone)]
pub struct Edges<A: Ord> {
    edges: Vec<A>,
}

impl<A: Ord> From<Vec<A>> for Edges<A> {

    /// Get an `Edges` instance from a `Vec<A>`:
    /// the vector will be sorted in increasing order
    /// using an unstable sorting algorithm and duplicates
    /// will be removed.
    ///
    /// # Example:
    ///
    /// ```
    /// extern crate ndarray_stats;
    /// #[macro_use(array)]
    /// extern crate ndarray;
    /// use ndarray_stats::histogram::Edges;
    ///
    /// # fn main() {
    /// let edges = Edges::from(array![1, 15, 10, 10, 20]);
    /// // The array gets sorted!
    /// assert_eq!(
    ///     edges[2],
    ///     15
    /// );
    /// # }
    /// ```
    fn from(mut edges: Vec<A>) -> Self {
        // sort the array in-place
        edges.sort_unstable();
        // remove duplicates
        edges.dedup();
        Edges { edges }
    }
}

impl<A: Ord + Clone> From<Array1<A>> for Edges<A> {
    /// Get an `Edges` instance from a `Array1<A>`:
    /// the array elements will be sorted in increasing order
    /// using an unstable sorting algorithm.
    ///
    /// # Example:
    ///
    /// ```
    /// extern crate ndarray_stats;
    /// use ndarray_stats::histogram::Edges;
    ///
    /// let edges = Edges::from(vec![1, 15, 10, 20]);
    /// // The vec gets sorted!
    /// assert_eq!(
    ///     edges[1],
    ///     10
    /// );
    /// ```
    fn from(edges: Array1<A>) -> Self {
        let edges = edges.to_vec();
        Self::from(edges)
    }
}

impl<A: Ord> Index<usize> for Edges<A>{
    type Output = A;

    /// Get the `i`-th edge.
    ///
    /// **Panics** if the index `i` is out of bounds.
    ///
    /// # Example:
    ///
    /// ```
    /// extern crate ndarray_stats;
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

impl<A: Ord> IntoIterator for Edges<A> {
    type Item = A;
    type IntoIter = ::std::vec::IntoIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        self.edges.into_iter()
    }
}

impl<A: Ord> Edges<A> {
    /// Number of edges in `self`.
    ///
    /// # Example:
    ///
    /// ```
    /// extern crate ndarray_stats;
    /// extern crate noisy_float;
    /// use ndarray_stats::histogram::Edges;
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(0.), n64(1.), n64(3.)]);
    /// assert_eq!(
    ///     edges.len(),
    ///     3
    /// );
    /// ```
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Borrow an immutable reference to the edges as a vector
    /// slice.
    ///
    /// # Example:
    ///
    /// ```
    /// extern crate ndarray_stats;
    /// use ndarray_stats::histogram::Edges;
    ///
    /// let edges = Edges::from(vec![0, 5, 3]);
    /// assert_eq!(
    ///     edges.as_slice(),
    ///     vec![0, 3, 5].as_slice()
    /// );
    /// ```
    pub fn as_slice(&self) -> &[A] {
        &self.edges
    }

    /// Given `value`, it returns an option:
    /// - `Some((left, right))`, where `right=left+1`, if there are two consecutive edges in
    /// `self` such that `self[left] <= value < self[right]`;
    /// - `None`, otherwise.
    ///
    /// # Example:
    ///
    /// ```
    /// extern crate ndarray_stats;
    /// use ndarray_stats::histogram::Edges;
    ///
    /// let edges = Edges::from(vec![0, 2, 3]);
    /// assert_eq!(
    ///     edges.indexes(&1),
    ///     Some((0, 1))
    /// );
    /// assert_eq!(
    ///     edges.indexes(&5),
    ///     None
    /// );
    /// ```
    pub fn indexes(&self, value: &A) -> Option<(usize, usize)> {
        // binary search for the correct bin
        let n_edges = self.len();
        match self.edges.binary_search(value) {
            Ok(i) if i == n_edges - 1 => None,
            Ok(i) => Some((i, i + 1)),
            Err(i) => {
                match i {
                    0 => None,
                    j if j == n_edges => None,
                    j => Some((j - 1, j)),
                }
            }
        }
    }
}

/// `Bins` is a sorted collection of non-overlapping
/// 1-dimensional intervals.
///
/// All intervals are left-inclusive and right-exclusive.
///
/// # Example:
///
/// ```
/// extern crate ndarray_stats;
/// extern crate noisy_float;
/// use ndarray_stats::histogram::{Edges, Bins};
/// use noisy_float::types::n64;
///
/// let edges = Edges::from(vec![n64(0.), n64(1.), n64(2.)]);
/// let bins = Bins::new(edges);
/// // first bin
/// assert_eq!(
///     bins.get(0),
///     n64(0.)..n64(1.) // n64(1.) is not included in the bin!
/// );
/// // second bin
/// assert_eq!(
///     bins.get(1),
///     n64(1.)..n64(2.)
/// );
/// ```
#[derive(Clone)]
pub struct Bins<A: Ord> {
    edges: Edges<A>,
}

impl<A: Ord> Bins<A> {
    /// Given a collection of [`Edges`], it returns the corresponding `Bins` instance.
    ///
    /// [`Edges`]: struct.Edges.html
    pub fn new(edges: Edges<A>) -> Self {
        Bins { edges }
    }

    /// Returns the number of bins.
    ///
    /// # Example:
    ///
    /// ```
    /// extern crate ndarray_stats;
    /// extern crate noisy_float;
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
    pub fn len(&self) -> usize {
        match self.edges.len() {
            0 => 0,
            n => n - 1,
        }
    }

    /// Given `value`, it returns:
    /// - `Some(i)`, if the `i`-th bin in `self` contains `value`;
    /// - `None`, if `value` does not belong to any of the bins in `self`.
    ///
    /// # Example:
    ///
    /// ```
    /// extern crate ndarray_stats;
    /// use ndarray_stats::histogram::{Edges, Bins};
    ///
    /// let edges = Edges::from(vec![0, 2, 4, 6]);
    /// let bins = Bins::new(edges);
    /// let value = 1;
    /// assert_eq!(
    ///     bins.index(&1),
    ///     Some(0)
    /// );
    /// assert_eq!(
    ///     bins.get(bins.index(&1).unwrap()),
    ///     0..2
    /// );
    /// ```
    pub fn index(&self, value: &A) -> Option<usize> {
        self.edges.indexes(value).map(|t| t.0)
    }

    /// Given `value`, it returns:
    /// - `Some(left_edge..right_edge)`, if there exists a bin in `self` such that
    ///  `left_edge <= value < right_edge`;
    /// - `None`, otherwise.
    ///
    /// # Example:
    ///
    /// ```
    /// extern crate ndarray_stats;
    /// use ndarray_stats::histogram::{Edges, Bins};
    ///
    /// let edges = Edges::from(vec![0, 2, 4, 6]);
    /// let bins = Bins::new(edges);
    /// assert_eq!(
    ///     bins.range(&1),
    ///     Some(0..2)
    /// );
    /// assert_eq!(
    ///     bins.range(&10),
    ///     None
    /// );
    /// ```
    pub fn range(&self, value: &A) -> Option<Range<A>>
        where
            A: Clone,
    {
        let edges_indexes = self.edges.indexes(value);
        edges_indexes.map(
            |t| {
                let (left, right) = t;
                Range {
                    start: self.edges[left].clone(),
                    end: self.edges[right].clone(),
                }
            }
        )
    }
}

impl<A: Ord + Clone> Bins<A> {
    /// Get the `i`-th bin.
    ///
    /// **Panics** if `index` is out of bounds.
    ///
    /// # Example:
    ///
    /// ```
    /// extern crate ndarray_stats;
    /// use ndarray_stats::histogram::{Edges, Bins};
    ///
    /// let edges = Edges::from(vec![1, 5, 10, 20]);
    /// let bins = Bins::new(edges);
    /// assert_eq!(
    ///     bins.get(1),
    ///     5..10
    /// );
    /// ```
    pub fn get(&self, index: usize) -> Range<A> {
        // It was not possible to implement this functionality
        // using the `Index` trait unless we were willing to
        // allocate a `Vec<Range<A>>` in the struct.
        // Index, in fact, forces you to return a reference.
        Range {
            start: self.edges[index].clone(),
            end: self.edges[index+1].clone(),
        }
    }
}

#[cfg(test)]
mod edges_tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::iter::FromIterator;

    quickcheck! {
        fn check_sorted_from_vec(v: Vec<i32>) -> bool {
            let edges = Edges::from(v);
            let n = edges.len();
            for i in 1..n {
                if edges[i-1] > edges[i] {
                    return false;
                }
            }
            true
        }

        fn check_sorted_from_array(v: Vec<i32>) -> bool {
            let a = Array1::from_vec(v);
            let edges = Edges::from(a);
            let n = edges.len();
            for i in 1..n {
                assert!(edges[i-1] <= edges[i]);
            }
            true
        }

        fn edges_are_right_exclusive(v: Vec<i32>) -> bool {
            let edges = Edges::from(v);
            let last = edges.as_slice().last();
            match last {
                None => true,
                Some(x) => {
                    edges.indexes(x).is_none()
                }
            }
        }

        fn edges_are_left_inclusive(v: Vec<i32>) -> bool {
            let edges = Edges::from(v);
            match edges.len() {
                1 => true,
                _ => {
                    let first = edges.as_slice().first();
                    match first {
                        None => true,
                        Some(x) => {
                            edges.indexes(x).is_some()
                        }
                    }
                }
            }
        }

        fn edges_are_deduped(v: Vec<i32>) -> bool {
            let unique_elements = BTreeSet::from_iter(v.iter());
            let edges = Edges::from(v.clone());
            let unique_edges = BTreeSet::from_iter(edges.as_slice().iter());
            unique_edges == unique_elements
        }
    }
}

#[cfg(test)]
mod bins_tests {
    use super::*;

    #[test]
    #[should_panic]
    fn get_panics_for_out_of_bound_indexes() {
        let edges = Edges::from(vec![0]);
        let bins = Bins::new(edges);
        // we need at least two edges to make a valid bin!
        bins.get(0);
    }
}
