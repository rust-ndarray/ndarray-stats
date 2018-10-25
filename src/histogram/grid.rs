use super::bins::Bins;

pub struct Grid<A: Ord> {
    grid: Vec<Bins<A>>,
}
