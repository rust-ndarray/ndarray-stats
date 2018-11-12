use super::bins::Bins;
use super::strategies::BinsBuildingStrategy;
use std::ops::Range;
use std::marker::PhantomData;
use itertools::izip;
use ndarray::{ArrayBase, Data, Ix1, Ix2, Axis};

/// A `Grid` is a partition of a rectangular region of an *n*-dimensional
/// space—e.g. [*a*<sub>0</sub>, *b*<sub>0</sub>) × ⋯ × [*a*<sub>*n*−1</sub>,
/// *b*<sub>*n*−1</sub>)—into a collection of rectangular *n*-dimensional bins.
///
/// The grid is **fully determined by its 1-dimensional projections** on the
/// coordinate axes. For example, this is a partition that can be represented
/// as a `Grid` struct:
/// ```text
/// +---+-------+-+
/// |   |       | |
/// +---+-------+-+
/// |   |       | |
/// |   |       | |
/// |   |       | |
/// |   |       | |
/// +---+-------+-+
/// ```
/// while the next one can't:
/// ```text
/// +---+-------+-+
/// |   |       | |
/// |   +-------+-+
/// |   |         |
/// |   |         |
/// |   |         |
/// |   |         |
/// +---+-------+-+
/// ```
///
/// # Example:
///
/// ```
/// extern crate ndarray_stats;
/// extern crate ndarray;
/// extern crate noisy_float;
/// use ndarray::{Array, array};
/// use ndarray_stats::HistogramExt;
/// use ndarray_stats::histogram::{Histogram, Grid, GridBuilder};
/// use ndarray_stats::histogram::strategies::Auto;
/// use noisy_float::types::{N64, n64};
///
/// # fn main() {
/// // 1-dimensional observations, as a (n_observations, 1) 2-d matrix
/// let observations = Array::from_shape_vec(
///     (12, 1),
///     vec![1, 4, 5, 2, 100, 20, 50, 65, 27, 40, 45, 23],
/// ).unwrap();
///
/// // The optimal grid layout is inferred from the data,
/// // specifying a strategy (Auto in this case)
/// let grid = GridBuilder::<usize, Auto<usize>>::from_array(&observations).build();
/// let histogram = observations.histogram(grid);
///
/// let histogram_matrix = histogram.as_view();
/// // Bins are left inclusive, right exclusive!
/// let expected = array![4, 1, 2, 1, 2, 0, 1, 0, 0, 1, 0, 0];
/// assert_eq!(histogram_matrix, expected.into_dyn());
/// # }
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Grid<A: Ord> {
    projections: Vec<Bins<A>>,
}

impl<A: Ord> From<Vec<Bins<A>>> for Grid<A> {

    /// Get a `Grid` instance from a `Vec<Bins<A>>`.
    ///
    /// The `i`-th element in `Vec<Bins<A>>` represents the 1-dimensional
    /// projection of the bin grid on the `i`-th axis.
    ///
    /// Alternatively, a `Grid` can be built directly from data using a
    /// [`GridBuilder`].
    ///
    /// [`GridBuilder`]: struct.GridBuilder.html
    fn from(projections: Vec<Bins<A>>) -> Self {
        Grid { projections }
    }
}

impl<A: Ord> Grid<A> {
    /// Returns `n`, the number of dimensions of the region partitioned by the grid.
    pub fn ndim(&self) -> usize {
        self.projections.len()
    }

    /// Returns the number of bins along each coordinate axis.
    pub fn shape(&self) -> Vec<usize> {
        self.projections.iter().map(|e| e.len()).collect()
    }

    /// Returns the grid projections on the coordinate axes as a slice of immutable references.
    pub fn projections(&self) -> &[Bins<A>] {
        &self.projections
    }

    /// Returns the index of the *n*-dimensional bin containing the point, if
    /// one exists.
    ///
    /// Returns `None` if the point is outside the grid.
    ///
    /// **Panics** if `point.len()` does not equal `self.ndim()`.
    pub fn index_of<S>(&self, point: &ArrayBase<S, Ix1>) -> Option<Vec<usize>>
    where
        S: Data<Elem = A>,
    {
        assert_eq!(point.len(), self.ndim(),
                   "Dimension mismatch: the point has {:?} dimensions, the grid \
                   expected {:?} dimensions.", point.len(), self.ndim());
        point
            .iter()
            .zip(self.projections.iter())
            .map(|(v, e)| e.index_of(v))
            .collect()
    }
}

impl<A: Ord + Clone> Grid<A> {
    /// Given `i=(i_0, ..., i_{n-1})`, an `n`-dimensional index, it returns
    /// `I_{i_0}x...xI_{i_{n-1}}`, an `n`-dimensional bin, where `I_{i_j}` is
    /// the `i_j`-th interval on the `j`-th projection of the grid on the coordinate axes.
    ///
    /// **Panics** if at least one among `(i_0, ..., i_{n-1})` is out of bounds on the respective
    /// coordinate axis - i.e. if there exists `j` such that `i_j >= self.projections[j].len()`.
    pub fn index(&self, index: &[usize]) -> Vec<Range<A>> {
        assert_eq!(index.len(), self.ndim(),
                   "Dimension mismatch: the index has {0:?} dimensions, the grid \
                   expected {1:?} dimensions.", index.len(), self.ndim());
        izip!(&self.projections, index)
            .map(|(bins, &i)| bins.index(i))
            .collect()
    }
}

/// `GridBuilder`, given a [`strategy`] and some observations, returns a [`Grid`]
/// instance for [`histogram`] computation.
///
/// [`Grid`]: struct.Grid.html
/// [`histogram`]: trait.HistogramExt.html
/// [`strategy`]: strategies/index.html
pub struct GridBuilder<A: Ord, B: BinsBuildingStrategy<A>> {
    bin_builders: Vec<B>,
    phantom: PhantomData<A>
}

impl<A: Ord, B: BinsBuildingStrategy<A>> GridBuilder<A, B> {
    /// Given some observations in a 2-dimensional array with shape `(n_observations, n_dimension)`
    /// it returns a `GridBuilder` instance that has learned the required parameter
    /// to build a [`Grid`] according to the specified [`strategy`].
    ///
    /// [`Grid`]: struct.Grid.html
    /// [`strategy`]: strategies/index.html
    pub fn from_array<S>(array: &ArrayBase<S, Ix2>) -> Self
    where
        S: Data<Elem=A>,
    {
        let bin_builders = array
            .axis_iter(Axis(1))
            .map(|data| B::from_array(data))
            .collect();
        Self { bin_builders, phantom: PhantomData }
    }

    /// Returns a [`Grid`] instance, built accordingly to the specified [`strategy`]
    /// using the parameters inferred from observations in [`from_array`].
    ///
    /// [`Grid`]: struct.Grid.html
    /// [`strategy`]: strategies/index.html
    /// [`from_array`]: #method.from_array.html
    pub fn build(&self) -> Grid<A> {
        let projections: Vec<_> = self.bin_builders.iter().map(|b| b.build()).collect();
        Grid::from(projections)
    }
}
