use super::bins::Bins;
use super::errors::BinNotFound;
use std::ops::Range;
use std::slice::Iter;
use ndarray::ArrayView1;

/// A `Grid` is a partition of a rectangular region of an `n`-dimensional
/// space (e.g. `[a_1, b_1]x...x[a_n, b_n]`) into a collection of
/// rectangular `n`-dimensional bins.
///
/// The grid is **fully determined by its 1-dimensional projections** on the
/// coordinate axes. For example, this is a partition that can be represented
/// as a `Grid` struct:
/// ```rust,ignore
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
/// ```rust,ignore
/// +---+-------+-+
/// |   |       | |
/// |   +-------+-+
/// |   |         |
/// |   |         |
/// |   |         |
/// |   |         |
/// +---+-------+-+
/// ```
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
    /// Returns `n`, the number of dimensions of the region partitioned by the grid.
    pub fn ndim(&self) -> usize {
        self.projections.len()
    }

    /// Returns `v=(v_i)_i`, a vector, where `v_i` is the number of bins in the grid projection
    /// on the `i`-th coordinate axis.
    pub fn shape(&self) -> Vec<usize> {
        self.projections.iter().map(|e| e.len()).collect::<Vec<_>>()
    }

    /// Returns the grid projections on the coordinate axes as a slice of immutable references.
    pub fn projections(&self) -> &[Bins<A>] {
        &self.projections
    }

    /// Given `P=(p_1, ..., p_n)`, a point, it returns:
    /// - `Ok(i)`, where `i=(i_1, ..., i_n)`, if `p_j` belongs to `i_j`-th bin
    /// on the `j`-th grid projection on the coordinate axes for all `j` in `{1, ..., n}`;
    /// - `Err(BinNotFound)`, if `P` does not belong to the region of space covered by the grid.
    pub fn index(&self, point: ArrayView1<A>) -> Result<Vec<usize>, BinNotFound> {
        assert_eq!(point.len(), self.ndim(),
                   "Dimension mismatch: the point has {0:?} dimensions, the grid \
                   expected {1:?} dimensions.", point.len(), self.ndim());
        point
            .iter()
            .zip(self.projections.iter())
            .map(|(v, e)| e.index(v).ok_or(BinNotFound))
            .collect::<Result<Vec<_>, _>>()
    }
}

impl<A: Ord + Clone> Grid<A> {
    /// Given `i=(i_1, ..., i_n)`, an `n`-dimensional index, it returns `I_{i_1}x...xI_{i_n}`, an
    /// `n`-dimensional bin, where `I_{i_j}` is the `i_j`-th interval on the `j`-th projection
    /// of the grid on the coordinate axes.
    pub fn get(&self, index: &[usize]) -> Vec<Range<A>> {
        assert_eq!(index.len(), self.ndim(),
                   "Dimension mismatch: the index has {0:?} dimensions, the grid \
                   expected {1:?} dimensions.", index.len(), self.ndim());
        let mut bin = vec![];
        for (axis_index, i) in index.iter().enumerate() {
            bin.push(self.projections[axis_index].get(*i));
        }
        bin
    }
}
