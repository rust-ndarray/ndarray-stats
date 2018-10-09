use ndarray::prelude::*;
use std::ops::Range;
use std::error;
use std::fmt;

#[derive(Debug, Clone)]
struct BinNotFound;

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

/// Wrapper around `Vec` that makes sure the elements are in ascending order.
struct Edges<A: Ord> {
    edges: Vec<A>,
}

impl<A: Ord + Clone> From<Array1<A>> for Edges<A> {
    fn from(edges: Array1<A>) -> Self {
        let mut edges = edges.to_vec();
        // sort the array in-place
        edges.sort_unstable();
        Edges { edges }
    }
}

impl<A: Ord> From<Vec<A>> for Edges<A> {
    fn from(mut edges: Vec<A>) -> Self {
        // sort the array in-place
        edges.sort_unstable();
        Edges { edges }
    }
}

impl<A: Ord> Edges<A> {
    fn len(&self) -> usize {
        self.edges.len()
    }

    fn slice(&self) -> &[A] {
        &self.edges
    }

    /// Returns the index of the bin containing the given value,
    /// or `None` if none of the bins contain the value.
    fn bin_index(&self, value: &A) -> Option<usize> {
        // binary search for the correct bin
        let n = self.len();
        match self.edges.binary_search(value) {
            Ok(i) => Some(i),
            Err(i) => {
                match i {
                    0 => None,
                    j if j == n => None,
                    _ => Some(i - 1),
                }
            }
        }
    }

    /// Returns the range of the bin containing the given value.
    fn bin_range(&self, value: &A) -> Option<Range<A>>
    where
        A: Clone,
    {
        let i = self.bin_index(value);
        match i {
            Some(j) => Some(
                Range { start: self.edges[j].clone(),
                        end: self.edges[j + 1].clone() }
            ),
            None => None,
        }
    }
}

struct HistogramCounts<A: Ord> {
    counts: ArrayD<usize>,
    edges: Vec<Edges<A>>,
}

struct HistogramDensity<A: Ord> {
    density: ArrayD<A>,
    edges: Vec<Edges<A>>,
}

impl<A: Ord> HistogramCounts<A> {
    pub fn new(edges: Vec<Edges<A>>) -> Self {
        let counts = ArrayD::zeros(edges.iter().map(|e| e.len() - 1).collect::<Vec<_>>());
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