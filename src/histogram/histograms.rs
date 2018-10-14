use ndarray::prelude::*;
use super::bins::Edges;
use super::errors::BinNotFound;

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
