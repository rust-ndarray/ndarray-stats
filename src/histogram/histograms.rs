use ndarray::prelude::*;
use ndarray::Data;
use super::grid::Grid;
use super::errors::BinNotFound;

/// Histogram data structure.
pub struct Histogram<A: Ord> {
    counts: ArrayD<usize>,
    grid: Grid<A>,
}

impl<A: Ord> Histogram<A> {
    /// Returns a new instance of Histogram given a [`Grid`].
    ///
    /// [`Grid`]: struct.Grid.html
    pub fn new(grid: Grid<A>) -> Self {
        let counts = ArrayD::zeros(grid.shape());
        Histogram { counts, grid }
    }

    /// Adds a single observation to the histogram.
    ///
    /// **Panics** if dimensions do not match: `self.ndim() != observation.len()`.
    ///
    /// # Example:
    /// ```
    /// extern crate ndarray_stats;
    /// #[macro_use(array)]
    /// extern crate ndarray;
    /// extern crate noisy_float;
    /// use ndarray_stats::histogram::{Edges, Bins, Histogram, Grid};
    /// use noisy_float::types::n64;
    ///
    /// # fn main() {
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut histogram = Histogram::new(square_grid);
    ///
    /// let observation = array![n64(0.5), n64(0.6)];
    ///
    /// histogram.add_observation(observation.view());
    ///
    /// let histogram_matrix = histogram.as_view();
    /// let expected = array![
    ///     [0, 0],
    ///     [0, 1],
    /// ];
    /// assert_eq!(histogram_matrix, expected.into_dyn());
    /// # }
    /// ```
    pub fn add_observation<S>(&mut self, observation: &ArrayBase<S, Ix1>) -> Result<(), BinNotFound>
    where
        S: Data<Elem = A>,
    {
        let bin_index = self.grid.index_of(observation)?;
        self.counts[&*bin_index] += 1;
        Ok(())
    }

    /// Returns the number of dimensions of the space the histogram is covering.
    pub fn ndim(&self) -> usize {
        debug_assert_eq!(self.counts.ndim(), self.grid.ndim());
        self.counts.ndim()
    }

    /// Borrows a view on the histogram matrix.
    pub fn as_view(&self) -> ArrayViewD<usize> {
        self.counts.view()
    }

    /// Borrows an immutable reference to the histogram grid.
    pub fn grid(&self) -> &Grid<A> {
        &self.grid
    }
}

/// Extension trait for `ArrayBase` providing methods to compute histograms.
pub trait HistogramExt<A, S>
    where
        S: Data<Elem = A>,
{
    /// Returns the [histogram](https://en.wikipedia.org/wiki/Histogram)
    /// for a 2-dimensional array of points `M`.
    ///
    /// Let `(n, d)` be the shape of `M`:
    /// - `n` is the number of points;
    /// - `d` is the number of dimensions of the space those points belong to.
    /// It follows that every column in `M` is a `d`-dimensional point.
    ///
    /// For example: a (3, 4) matrix `M` is a collection of 3 points in a
    /// 4-dimensional space.
    ///
    /// Important: points outside the grid are ignored!
    ///
    /// **Panics** if `d` is different from `grid.ndim()`.
    ///
    /// # Example:
    ///
    /// ```
    /// extern crate ndarray_stats;
    /// #[macro_use(array)]
    /// extern crate ndarray;
    /// extern crate noisy_float;
    /// use ndarray_stats::HistogramExt;
    /// use ndarray_stats::histogram::{Histogram, Grid, GridBuilder, strategies::Sqrt};
    /// use noisy_float::types::{N64, n64};
    ///
    /// # fn main() {
    /// let observations = array![
    ///     [n64(1.), n64(0.5)],
    ///     [n64(-0.5), n64(1.)],
    ///     [n64(-1.), n64(-0.5)],
    ///     [n64(0.5), n64(-1.)]
    /// ];
    /// let grid = GridBuilder::<N64, Sqrt<N64>>::from_array(&observations).build();
    /// let histogram = observations.histogram(grid);
    ///
    /// let histogram_matrix = histogram.as_view();
    /// // Bins are left inclusive, right exclusive!
    /// let expected = array![
    ///     [1, 0],
    ///     [1, 0],
    /// ];
    /// assert_eq!(histogram_matrix, expected.into_dyn());
    /// # }
    /// ```
    fn histogram(&self, grid: Grid<A>) -> Histogram<A>
        where
            A: Ord;
}

impl<A, S> HistogramExt<A, S> for ArrayBase<S, Ix2>
    where
        S: Data<Elem = A>,
        A: Ord,
{
    fn histogram(&self, grid: Grid<A>) -> Histogram<A>
    {
        let mut histogram = Histogram::new(grid);
        for point in self.axis_iter(Axis(0)) {
            histogram.add_observation(&point);
        }
        histogram
    }
}
