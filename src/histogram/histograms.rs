use super::errors::BinNotFound;
use super::grid::Grid;
use ndarray::prelude::*;
use ndarray::Data;

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
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{Edges, Bins, Histogram, Grid};
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut histogram = Histogram::new(square_grid);
    ///
    /// let observation = array![n64(0.5), n64(0.6)];
    ///
    /// histogram.add_observation(&observation)?;
    ///
    /// let histogram_matrix = histogram.counts();
    /// let expected = array![
    ///     [0, 0],
    ///     [0, 1],
    /// ];
    /// assert_eq!(histogram_matrix, expected.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn add_observation<S>(&mut self, observation: &ArrayBase<S, Ix1>) -> Result<(), BinNotFound>
    where
        S: Data<Elem = A>,
    {
        match self.grid.index_of(observation) {
            Some(bin_index) => {
                self.counts[&*bin_index] += 1;
                Ok(())
            }
            None => Err(BinNotFound),
        }
    }

    /// Returns the number of dimensions of the space the histogram is covering.
    pub fn ndim(&self) -> usize {
        debug_assert_eq!(self.counts.ndim(), self.grid.ndim());
        self.counts.ndim()
    }

    /// Borrows a view on the histogram counts matrix.
    pub fn counts(&self) -> ArrayViewD<'_, usize> {
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
    /// use ndarray::array;
    /// use ndarray_stats::{
    ///     HistogramExt,
    ///     histogram::{
    ///         Histogram, Grid, GridBuilder,
    ///         Edges, Bins,
    ///         strategies::Sqrt},
    /// };
    /// use noisy_float::types::{N64, n64};
    ///
    /// let observations = array![
    ///     [n64(1.), n64(0.5)],
    ///     [n64(-0.5), n64(1.)],
    ///     [n64(-1.), n64(-0.5)],
    ///     [n64(0.5), n64(-1.)]
    /// ];
    /// let grid = GridBuilder::<Sqrt<N64>>::from_array(&observations).unwrap().build();
    /// let expected_grid = Grid::from(
    ///     vec![
    ///         Bins::new(Edges::from(vec![n64(-1.), n64(0.), n64(1.), n64(2.)])),
    ///         Bins::new(Edges::from(vec![n64(-1.), n64(0.), n64(1.), n64(2.)])),
    ///     ]
    /// );
    /// assert_eq!(grid, expected_grid);
    ///
    /// let histogram = observations.histogram(grid);
    ///
    /// let histogram_matrix = histogram.counts();
    /// // Bins are left inclusive, right exclusive!
    /// let expected = array![
    ///     [1, 0, 1],
    ///     [1, 0, 0],
    ///     [0, 1, 0],
    /// ];
    /// assert_eq!(histogram_matrix, expected.into_dyn());
    /// ```
    fn histogram(&self, grid: Grid<A>) -> Histogram<A>
    where
        A: Ord;

    private_decl! {}
}

impl<A, S> HistogramExt<A, S> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    A: Ord,
{
    fn histogram(&self, grid: Grid<A>) -> Histogram<A> {
        let mut histogram = Histogram::new(grid);
        for point in self.axis_iter(Axis(0)) {
            let _ = histogram.add_observation(&point);
        }
        histogram
    }

    private_impl! {}
}
