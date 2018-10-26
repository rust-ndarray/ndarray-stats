use super::bins::Bins;
use super::errors::BinNotFound;
use std::slice::Iter;
use ndarray::ArrayView1;

pub struct Grid<A: Ord> {
    projections: Vec<Bins<A>>,
}

impl<A: Ord> From<Vec<Bins<A>>> for Grid<A> {

    /// Get a `Grid` instance from a `Vec<Bins<A>>`.
    ///
    /// The `i`-th element in `Vec<Bins<A>>` represents the 1-dimensional
    /// projection of the bin grid on the `i`-th axis.
    fn from(projections: Vec<Bins<A>>) -> Self {
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

    pub fn shape(&self) -> Vec<usize> {
        self.iter_projections().map(|e| e.len()).collect::<Vec<_>>()
    }

    pub fn projections(&self) -> &[Bins<A>] {
        &self.projections
    }

    pub fn index(&self, point: ArrayView1<A>) -> Result<Vec<usize>, BinNotFound> {
        assert_eq!(point.len(), self.ndim(),
                   "Dimension mismatch: the point has {0:?} dimensions, the grid \
                   expected {1:?} dimensions.", point.len(), self.ndim());
        point
            .iter()
            .zip(self.iter_projections())
            .map(|(v, e)| e.index(v).ok_or(BinNotFound))
            .collect::<Result<Vec<_>, _>>()
    }
}
