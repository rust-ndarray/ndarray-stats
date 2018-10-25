use super::bins::Bins;
use std::slice::Iter;

pub struct Grid<A: Ord> {
    grid: Vec<Bins<A>>,
}

impl<A: Ord> Grid<A> {
    pub fn iter(&self) -> Iter<Bins<A>> {
        self.grid.iter()
    }

    pub fn ndim(&self) -> usize {
        self.grid.len()
    }
}
