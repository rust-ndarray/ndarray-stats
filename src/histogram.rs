use ndarray::prelude::*;
use std::ops::Range;
use std::error;
use std::fmt;

#[derive(Debug, Clone)]
pub struct BinNotFound;

impl fmt::Display for BinNotFound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "No bin has been found.")
    }
}

impl error::Error for BinNotFound {
    fn description(&self) -> &str {
        "No bin has been found."
    }

    fn cause(&self) -> Option<&error::Error> {
        // Generic error, underlying cause isn't tracked.
        None
    }
}

/// `Edges` is a sorted collection of `A` elements used
/// to represent the boundaries of intervals on
/// a 1-dimensional axis.
///
/// # Example:
///
/// ```
/// extern crate ndarray_stats;
/// extern crate noisy_float;
/// use ndarray_stats::Edges;
/// use noisy_float::types::n64;
///
/// let unit_interval = Edges::from(vec![n64(0.), n64(1.)]);
/// // left inclusive
/// assert_eq!(
///     unit_interval.bin_range(&n64(0.)).unwrap(),
///     n64(0.)..n64(1.),
/// );
/// // right exclusive
/// assert_eq!(
///     unit_interval.bin_range(&n64(1.)),
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
        let mut edges = edges.to_vec();
        Self::from(edges)
    }
}

impl<A: Ord> Edges<A> {
    pub fn n_intervals(&self) -> usize {
        match self.n_edges() {
            0 => 0,
            n => n - 1,
        }
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn slice(&self) -> &[A] {
        &self.edges
    }

    /// Returns the index of the bin containing the given value,
    /// or `None` if none of the bins contain the value.
    fn edges_indexes(&self, value: &A) -> Option<(usize, usize)> {
        // binary search for the correct bin
        let n_edges = self.n_edges();
        match self.edges.binary_search(value) {
            Ok(i) if i == n_edges-1 => None,
            Ok(i) => Some((i, i+1)),
            Err(i) => {
                match i {
                    0 => None,
                    j if j == n_edges => None,
                    j => Some((j-1, j)),
                }
            }
        }
    }

    /// Returns the index of the bin containing the given value,
    /// or `None` if none of the bins contain the value.
    fn bin_index(&self, value: &A) -> Option<usize> {
        self.edges_indexes(value).map(|t| t.0)
    }

    /// Returns the range of the bin containing the given value.
    pub fn bin_range(&self, value: &A) -> Option<Range<A>>
    where
        A: Clone,
    {
        let edges_indexes= self.edges_indexes(value);
        edges_indexes.map(
            |t| {
                let (left, right) = t;
                Range {
                    start: self.edges[left].clone(),
                    end: self.edges[right].clone()
                }
            }
        )
    }
}

pub struct HistogramCounts<A: Ord> {
    counts: ArrayD<usize>,
    edges: Vec<Edges<A>>,
}

struct HistogramDensity<A: Ord> {
    density: ArrayD<A>,
    edges: Vec<Edges<A>>,
}

impl<A: Ord> HistogramCounts<A> {
    pub fn new(edges: Vec<Edges<A>>) -> Self {
        let counts = ArrayD::zeros(edges.iter().map(|e| e.n_intervals()).collect::<Vec<_>>());
        HistogramCounts { counts, edges }
    }

    pub fn add_observation(&mut self, observation: ArrayView1<A>) -> Result<(), BinNotFound> {
        let bin = observation
            .iter()
            .zip(&self.edges)
            .map(|(v, e)| e.bin_index(v).ok_or(BinNotFound))
            .collect::<Result<Vec<_>, _>>()?;
        self.counts[IxDyn(&bin)] += 1;
        Ok(())
    }
}