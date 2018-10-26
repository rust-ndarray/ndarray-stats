use super::bins::Bins;
use super::errors::BinNotFound;
use std::slice::Iter;
use ndarray::ArrayView1;

pub struct Grid<A: Ord> {
    projections: Vec<Bins<A>>,
}

impl<A: Ord> From<Vec<Bins<A>>> for Grid<A> {

    /// Get a `Grid` instance from a `Vec<Bins<A>>`.
    fn from(mut projections: Vec<Bins<A>>) -> Self {
        Grid { projections }
    }
}

impl<A: Ord> Grid<A> {
    pub fn iter_projections(&self) -> Iter<Bins<A>> {
        self.projections.iter()
    }

    pub fn ndim(&self) -> usize {
        self.projections.len()
    }

    pub fn projections(&self) -> &[Bins<A>] {
        &self.projections
    }

    pub fn index(&self, observation: ArrayView1<A>) -> Result<Vec<usize>, BinNotFound> {
        assert_eq!(observation.len(), self.ndim(),
                   "Dimension mismatch: the observation has {0:?} dimensions, the grid \
                   instead has {1:?} dimensions.", observation.len(), self.ndim());
        observation
            .iter()
            .zip(self.grid.iter_projections())
            .map(|(v, e)| e.index(v).ok_or(BinNotFound))
            .collect::<Result<Vec<_>, _>>()?
    }
}
