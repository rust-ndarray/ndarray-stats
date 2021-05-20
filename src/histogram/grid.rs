#![warn(missing_docs, clippy::all, clippy::pedantic)]

use super::{bins::Bins, errors::BinsBuildError, strategies::BinsBuildingStrategy};
use itertools::izip;
use ndarray::{ArrayBase, Axis, Data, Ix1, Ix2};
use std::ops::Range;

/// An orthogonal partition of a rectangular region in an *n*-dimensional space, e.g.
/// [*a*<sub>0</sub>, *b*<sub>0</sub>) × ⋯ × [*a*<sub>*n*−1</sub>, *b*<sub>*n*−1</sub>),
/// represented as a collection of rectangular *n*-dimensional bins.
///
/// The grid is **solely determined by the Cartesian product of its projections** on each coordinate
/// axis. Therefore, each element in the product set should correspond to a sub-region in the grid.
///
/// For example, this partition can be represented as a `Grid` struct:
///
/// ```text
///
/// g +---+-------+---+
///   | 3 |   4   | 5 |
/// f +---+-------+---+
///   |   |       |   |
///   | 0 |   1   | 2 |
///   |   |       |   |
/// e +---+-------+---+
///   a   b       c   d
///
/// R0:    [a, b) × [e, f)
/// R1:    [b, c) × [e, f)
/// R2:    [c, d) × [e, f)
/// R3:    [a, b) × [f, g)
/// R4:    [b, d) × [f, g)
/// R5:    [c, d) × [f, g)
/// Grid:  { [a, b), [b, c), [c, d) } × { [e, f), [f, g) } == { R0, R1, R2, R3, R4, R5 }
/// ```
///
/// while the next one can't:
///
/// ```text
///  g  +---+-----+---+
///     |   |  2  | 3 |
/// (f) |   +-----+---+
///     | 0 |         |
///     |   |    1    |
///     |   |         |
///  e  +---+-----+---+
///     a   b     c   d
///
/// R0:    [a, b) × [e, g)
/// R1:    [b, d) × [e, f)
/// R2:    [b, c) × [f, g)
/// R3:    [c, d) × [f, g)
/// // 'f', as long as 'R1', 'R2', or 'R3', doesn't appear on LHS
/// // [b, c) × [e, g), [c, d) × [e, g) doesn't appear on RHS
/// Grid:  { [a, b), [b, c), [c, d) } × { [e, g) } != { R0, R1, R2, R3 }
/// ```
///
/// # Examples
///
/// Basic usage, building a `Grid` via [`GridBuilder`], with optimal grid layout determined by
/// a given [`strategy`], and generating a [`histogram`]:
///
/// ```
/// use ndarray::{Array, array};
/// use ndarray_stats::{
///     histogram::{strategies::Auto, Bins, Edges, Grid, GridBuilder},
///     HistogramExt,
/// };
///
/// // 1-dimensional observations, as a (n_observations, n_dimension) 2-d matrix
/// let observations = Array::from_shape_vec(
///     (12, 1),
///     vec![1, 4, 5, 2, 100, 20, 50, 65, 27, 40, 45, 23],
/// ).unwrap();
///
/// // The optimal grid layout is inferred from the data, given a chosen strategy, Auto in this case
/// let grid = GridBuilder::<Auto<usize>>::from_array(&observations).unwrap().build();
///
/// let histogram = observations.histogram(grid);
///
/// let histogram_matrix = histogram.counts();
/// // Bins are left-closed, right-open!
/// let expected = array![4, 3, 3, 1, 0, 1];
/// assert_eq!(histogram_matrix, expected.into_dyn());
/// ```
///
/// [`histogram`]: trait.HistogramExt.html
/// [`GridBuilder`]: struct.GridBuilder.html
/// [`strategy`]: strategies/index.html
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Grid<A: Ord> {
    projections: Vec<Bins<A>>,
}

impl<A: Ord> From<Vec<Bins<A>>> for Grid<A> {
    /// Converts a `Vec<Bins<A>>` into a `Grid<A>`, consuming the vector of bins.
    ///
    /// The `i`-th element in `Vec<Bins<A>>` represents the projection of the bin grid onto the
    /// `i`-th axis.
    ///
    /// Alternatively, a `Grid` can be built directly from data using a [`GridBuilder`].
    ///
    /// [`GridBuilder`]: struct.GridBuilder.html
    fn from(projections: Vec<Bins<A>>) -> Self {
        Grid { projections }
    }
}

impl<A: Ord> Grid<A> {
    /// Returns the number of dimensions of the region partitioned by the grid.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::{Edges, Bins, Grid};
    ///
    /// let edges = Edges::from(vec![0, 1]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    ///
    /// assert_eq!(square_grid.ndim(), 2usize)
    /// ```
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.projections.len()
    }

    /// Returns the numbers of bins along each coordinate axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray_stats::histogram::{Edges, Bins, Grid};
    ///
    /// let edges_x = Edges::from(vec![0, 1]);
    /// let edges_y = Edges::from(vec![-1, 0, 1]);
    /// let bins_x = Bins::new(edges_x);
    /// let bins_y = Bins::new(edges_y);
    /// let square_grid = Grid::from(vec![bins_x, bins_y]);
    ///
    /// assert_eq!(square_grid.shape(), vec![1usize, 2usize]);
    /// ```
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.projections.iter().map(Bins::len).collect()
    }

    /// Returns the grid projections on each coordinate axis as a slice of immutable references.
    #[must_use]
    pub fn projections(&self) -> &[Bins<A>] {
        &self.projections
    }

    /// Returns an `n-dimensional` index, of bins along each axis that contains the point, if one
    /// exists.
    ///
    /// Returns `None` if the point is outside the grid.
    ///
    /// # Panics
    ///
    /// Panics if dimensionality of the point doesn't equal the grid's.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{Edges, Bins, Grid};
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    ///
    /// // (0., -0.7) falls in 1st and 0th bin respectively
    /// assert_eq!(
    ///     square_grid.index_of(&array![n64(0.), n64(-0.7)]),
    ///     Some(vec![1, 0]),
    /// );
    /// // Returns `None`, as `1.` is outside the grid since bins are right-open
    /// assert_eq!(
    ///     square_grid.index_of(&array![n64(0.), n64(1.)]),
    ///     None,
    /// );
    /// ```
    ///
    /// A panic upon dimensionality mismatch:
    ///
    /// ```should_panic
    /// # use ndarray::array;
    /// # use ndarray_stats::histogram::{Edges, Bins, Grid};
    /// # use noisy_float::types::n64;
    /// # let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// # let bins = Bins::new(edges);
    /// # let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// // the point has 3 dimensions, the grid expected 2 dimensions
    /// assert_eq!(
    ///     square_grid.index_of(&array![n64(0.), n64(-0.7), n64(0.5)]),
    ///     Some(vec![1, 0, 1]),
    /// );
    /// ```
    pub fn index_of<S>(&self, point: &ArrayBase<S, Ix1>) -> Option<Vec<usize>>
    where
        S: Data<Elem = A>,
    {
        assert_eq!(
            point.len(),
            self.ndim(),
            "Dimension mismatch: the point has {:?} dimensions, the grid \
             expected {:?} dimensions.",
            point.len(),
            self.ndim()
        );
        point
            .iter()
            .zip(self.projections.iter())
            .map(|(v, e)| e.index_of(v))
            .collect()
    }
}

impl<A: Ord + Clone> Grid<A> {
    /// Given an `n`-dimensional index, `i = (i_0, ..., i_{n-1})`, returns an `n`-dimensional bin,
    /// `I_{i_0} x ... x I_{i_{n-1}}`, where `I_{i_j}` is the `i_j`-th interval on the `j`-th
    /// projection of the grid on the coordinate axes.
    ///
    /// # Panics
    ///
    /// Panics if at least one in the index, `(i_0, ..., i_{n-1})`, is out of bounds on the
    /// corresponding coordinate axis, i.e. if there exists `j` s.t.
    /// `i_j >= self.projections[j].len()`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{Edges, Bins, Grid};
    ///
    /// let edges_x = Edges::from(vec![0, 1]);
    /// let edges_y = Edges::from(vec![2, 3, 4]);
    /// let bins_x = Bins::new(edges_x);
    /// let bins_y = Bins::new(edges_y);
    /// let square_grid = Grid::from(vec![bins_x, bins_y]);
    ///
    /// // Query the 0-th bin on x-axis, and 1-st bin on y-axis
    /// assert_eq!(
    ///     square_grid.index(&[0, 1]),
    ///     vec![0..1, 3..4],
    /// );
    /// ```
    ///
    /// A panic upon out-of-bounds:
    ///
    /// ```should_panic
    /// # use ndarray::array;
    /// # use ndarray_stats::histogram::{Edges, Bins, Grid};
    /// # let edges_x = Edges::from(vec![0, 1]);
    /// # let edges_y = Edges::from(vec![2, 3, 4]);
    /// # let bins_x = Bins::new(edges_x);
    /// # let bins_y = Bins::new(edges_y);
    /// # let square_grid = Grid::from(vec![bins_x, bins_y]);
    /// // out-of-bound on y-axis
    /// assert_eq!(
    ///     square_grid.index(&[0, 2]),
    ///     vec![0..1, 3..4],
    /// );
    /// ```
    #[must_use]
    pub fn index(&self, index: &[usize]) -> Vec<Range<A>> {
        assert_eq!(
            index.len(),
            self.ndim(),
            "Dimension mismatch: the index has {0:?} dimensions, the grid \
             expected {1:?} dimensions.",
            index.len(),
            self.ndim()
        );
        izip!(&self.projections, index)
            .map(|(bins, &i)| bins.index(i))
            .collect()
    }
}

/// A builder used to create [`Grid`] instances for [`histogram`] computations.
///
/// # Examples
///
/// Basic usage, creating a `Grid` with some observations and a given [`strategy`]:
///
/// ```
/// use ndarray::Array;
/// use ndarray_stats::histogram::{strategies::Auto, Bins, Edges, Grid, GridBuilder};
///
/// // 1-dimensional observations, as a (n_observations, n_dimension) 2-d matrix
/// let observations = Array::from_shape_vec(
///     (12, 1),
///     vec![1, 4, 5, 2, 100, 20, 50, 65, 27, 40, 45, 23],
/// ).unwrap();
///
/// // The optimal grid layout is inferred from the data, given a chosen strategy, Auto in this case
/// let grid = GridBuilder::<Auto<usize>>::from_array(&observations).unwrap().build();
/// // Equivalently, build a Grid directly
/// let expected_grid = Grid::from(vec![Bins::new(Edges::from(vec![1, 20, 39, 58, 77, 96, 115]))]);
///
/// assert_eq!(grid, expected_grid);
/// ```
///
/// [`Grid`]: struct.Grid.html
/// [`histogram`]: trait.HistogramExt.html
/// [`strategy`]: strategies/index.html
#[allow(clippy::module_name_repetitions)]
pub struct GridBuilder<B: BinsBuildingStrategy> {
    bin_builders: Vec<B>,
}

impl<A, B> GridBuilder<B>
where
    A: Ord,
    B: BinsBuildingStrategy<Elem = A>,
{
    /// Returns a `GridBuilder` for building a [`Grid`] with a given [`strategy`] and some
    /// observations in a 2-dimensionalarray with shape `(n_observations, n_dimension)`.
    ///
    /// # Errors
    ///
    /// It returns [`BinsBuildError`] if it is not possible to build a [`Grid`] given
    /// the observed data according to the chosen [`strategy`].
    ///
    /// # Examples
    ///
    /// See [Trait-level examples] for basic usage.
    ///
    /// [`Grid`]: struct.Grid.html
    /// [`strategy`]: strategies/index.html
    /// [`BinsBuildError`]: errors/enum.BinsBuildError.html
    /// [Trait-level examples]: struct.GridBuilder.html#examples
    pub fn from_array<S>(array: &ArrayBase<S, Ix2>) -> Result<Self, BinsBuildError>
    where
        S: Data<Elem = A>,
    {
        let bin_builders = array
            .axis_iter(Axis(1))
            .map(|data| B::from_array(&data))
            .collect::<Result<Vec<B>, BinsBuildError>>()?;
        Ok(Self { bin_builders })
    }

    /// Returns a [`Grid`] instance, with building parameters infered in [`from_array`], according
    /// to the specified [`strategy`] and observations provided.
    ///
    /// # Examples
    ///
    /// See [Trait-level examples] for basic usage.
    ///
    /// [`Grid`]: struct.Grid.html
    /// [`strategy`]: strategies/index.html
    /// [`from_array`]: #method.from_array.html
    #[must_use]
    pub fn build(&self) -> Grid<A> {
        let projections: Vec<_> = self.bin_builders.iter().map(|b| b.build()).collect();
        Grid::from(projections)
    }
}
