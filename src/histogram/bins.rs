use ndarray::prelude::*;
use std::ops::{Index, Range};

/// `Edges` is a sorted collection of `A` elements used
/// to represent the boundaries of intervals on
/// a 1-dimensional axis.
///
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
pub struct Edges<A: Ord> {
    edges: Vec<A>,
}

impl<A: Ord> From<Vec<A>> for Edges<A> {
    fn from(mut edges: Vec<A>) -> Self {
        // sort the array in-place
        edges.sort_unstable();
        Edges { edges }
    }
}

impl<A: Ord + Clone> From<Array1<A>> for Edges<A> {
    fn from(edges: Array1<A>) -> Self {
        let edges = edges.to_vec();
        Self::from(edges)
    }
}

impl<A: Ord> Index<usize> for Edges<A>{
    type Output = A;

    fn index(&self, i: usize) -> &A {
        &self.edges[i]
    }
}

impl<A: Ord> Edges<A> {
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn slice(&self) -> &[A] {
        &self.edges
    }

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

pub struct Bins<A: Ord> {
    edges: Edges<A>,
}

impl<A: Ord> Bins<A> {
    pub fn new(edges: Edges<A>) -> Self {
        Bins { edges }
    }

    pub fn len(&self) -> usize {
        match self.edges.len() {
            0 => 0,
            n => n - 1,
        }
    }

    pub fn index(&self, value: &A) -> Option<usize> {
        self.edges.indexes(value).map(|t| t.0)
    }

    /// Returns the range of the bin containing the given value.
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
    pub fn get(&self, index: usize) -> Range<A> {
        Range {
            start: self.edges[index].clone(),
            end: self.edges[index+1].clone(),
        }
    }
}
