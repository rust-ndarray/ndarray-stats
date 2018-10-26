use super::bins::Bins;
use std::slice::Iter;

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
    pub fn iter(&self) -> Iter<Bins<A>> {
        self.projections.iter()
    }

    pub fn ndim(&self) -> usize {
        self.projections.len()
    }
}
