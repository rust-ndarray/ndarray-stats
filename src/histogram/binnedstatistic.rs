use super::errors::BinNotFound;
use super::grid::Grid;
use ndarray::prelude::{ArrayBase, ArrayD, ArrayViewD, Axis, Ix1, Ix2};
use ndarray::{Data, Zip};
use num_traits::identities::{One, Zero};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
/// Binned statistic data structure.
pub struct BinnedStatistic<A: Ord, T: num_traits::Num> {
    counts: ArrayD<usize>,
    sum: ArrayD<T>,
    grid: Grid<A>,
}

impl<A, T> BinnedStatistic<A, T>
where
    A: Ord,
    T: Copy + num_traits::Num,
{
    /// Returns a new instance of BinnedStatistic given a [`Grid`].
    ///
    /// [`Grid`]: struct.Grid.html
    pub fn new(grid: Grid<A>) -> Self {
        let counts = ArrayD::zeros(grid.shape());
        let sum = ArrayD::zeros(grid.shape());
        BinnedStatistic { counts, sum, grid }
    }

    /// Adds a single sample to the binned statistic.
    ///
    /// **Panics** if dimensions do not match: `self.ndim() != sample.len()`.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{
    /// BinContent::Empty, BinContent::Value, BinnedStatistic, Bins, Edges, Grid,
    /// };
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, n64(1.0))?;
    /// binned_statistic.add_sample(&sample, n64(2.0))?;
    ///
    /// let binned_statistic_sum = binned_statistic.sum();
    /// let expected = array![
    ///     [0.0, 0.0],
    ///     [0.0, 3.0],
    /// ];
    /// assert_eq!(binned_statistic_sum, expected.into_dyn());
    ///
    /// let binned_statistic_bc = binned_statistic.sum_binned();
    /// let expected_value = array![
    ///     [Empty, Empty],
    ///     [Empty, Value(n64(3.0))],
    /// ];
    /// assert_eq!(binned_statistic_bc, expected_value.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn add_sample<S>(&mut self, sample: &ArrayBase<S, Ix1>, value: T) -> Result<(), BinNotFound>
    where
        S: Data<Elem = A>,
        T: Copy + num_traits::Num,
    {
        match self.grid.index_of(sample) {
            Some(bin_index) => {
                self.counts[&*bin_index] += 1usize;
                self.sum[&*bin_index] = self.sum[&*bin_index] + value;
                Ok(())
            }
            None => Err(BinNotFound),
        }
    }

    /// Returns the number of dimensions of the space the binned statistic is covering.
    pub fn ndim(&self) -> usize {
        debug_assert_eq!(self.counts.ndim(), self.grid.ndim());
        self.counts.ndim()
    }

    /// Borrows a view on the binned statistic `sum` matrix.
    pub fn sum(&self) -> ArrayViewD<'_, T> {
        self.sum.view()
    }

    /// Borrows a view on the binned statistic `counts` matrix (equivalent to histogram).
    pub fn counts(&self) -> ArrayViewD<'_, usize> {
        self.counts.view()
    }

    /// Borrows an immutable reference to the binned statistic grid.
    pub fn grid(&self) -> &Grid<A> {
        &self.grid
    }

    /// Returns an array of `BinContent`s of the `counts` matrix (equivalent to histogram).
    pub fn counts_binned(&self) -> ArrayD<BinContent<usize>> {
        let mut counts_binned = ArrayD::<BinContent<usize>>::zeros(self.counts.shape());

        for (counts_arr, binned) in self.counts.iter().zip(&mut counts_binned) {
            *binned = if *counts_arr == 0usize {
                BinContent::Empty
            } else {
                BinContent::Value(*counts_arr)
            };
        }
        counts_binned
    }

    /// Returns an array of `BinContents`s of the `sum` matrix.
    pub fn sum_binned(&self) -> ArrayD<BinContent<T>> {
        let mut sum_binned = ArrayD::<BinContent<T>>::zeros(self.counts.shape());

        Zip::from(&mut sum_binned)
            .and(&self.sum)
            .and(&self.counts)
            .apply(|w, &x, &y| {
                *w = if y == 0usize {
                    BinContent::Empty
                } else {
                    BinContent::Value(x)
                }
            });

        sum_binned
    }
}

impl<A: Ord, T: Copy + num_traits::Num + Add<Output = T>> Add for BinnedStatistic<A, T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        if self.grid != other.grid {
            panic!("`BinnedStatistics` can only be added for the same `grid`!")
        };

        BinnedStatistic {
            counts: &self.counts + &other.counts,
            sum: &self.sum + &other.sum,
            grid: self.grid,
        }
    }
}

/// Extension trait for `ArrayBase` providing methods to compute binned statistics.
pub trait BinnedStatisticExt<A, S, T>
where
    S: Data<Elem = A>,
    T: Copy + num_traits::Num,
{
    /// Returns the binned statistic for a 2-dimensional array of samples `M`
    /// and a 1-dimensional vector of values `N`.
    ///
    /// Let `(n, d)` be the shape of `M` and `(n)` the shape of `N`:
    /// - `n` is the number of samples/values;
    /// - `d` is the number of dimensions of the space those points belong to.
    /// It follows that every column in `M` is a `d`-dimensional sample
    /// and every value in `N` is the corresponding value.
    ///
    /// For example: a (3, 4) matrix `M` is a collection of 3 points in a
    /// 4-dimensional space with a corresponding (4) vector `N`.
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
    ///     BinnedStatisticExt,
    ///     histogram::{BinnedStatistic, Grid, Edges, Bins},
    /// };
    /// use noisy_float::types::{N64, n64};
    ///
    /// let samples = array![
    ///     [n64(1.5), n64(0.5)],
    ///     [n64(-0.5), n64(1.5)],
    ///     [n64(-1.), n64(-0.5)],
    ///     [n64(0.5), n64(-1.)]
    /// ];
    /// let values = array![n64(12.), n64(-0.5), n64(1.), n64(2.)].into_dyn();
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.), n64(2.)]);
    /// let bins = Bins::new(edges);
    /// let grid = Grid::from(vec![bins.clone(), bins.clone()]);
    ///
    /// let binned_statistic = samples.binned_statistic(grid, values);
    ///
    /// // Bins are left inclusive, right exclusive!
    /// let expected_counts = array![
    ///     [1, 0, 1],
    ///     [1, 0, 0],
    ///     [0, 1, 0]
    /// ];
    /// let expected_sum = array![
    ///     [n64(1.),  n64(0.),  n64(-0.5)],
    ///     [n64(2.),  n64(0.),  n64(0.)],
    ///     [n64(0.), n64(12.), n64(0.)]
    /// ];
    /// assert_eq!(binned_statistic.counts(), expected_counts.into_dyn());
    /// assert_eq!(binned_statistic.sum(), expected_sum.into_dyn());
    /// ```
    fn binned_statistic(&self, grid: Grid<A>, values: ArrayD<T>) -> BinnedStatistic<A, T>
    where
        A: Ord;

    private_decl! {}
}

/// Implementation of `BinnedStatisticExt` for `ArrayBase<S, Ix2>`.
impl<A, S, T> BinnedStatisticExt<A, S, T> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    A: Ord,
    T: Copy + num_traits::Num,
{
    fn binned_statistic(&self, grid: Grid<A>, values: ArrayD<T>) -> BinnedStatistic<A, T> {
        let mut binned_statistic = BinnedStatistic::new(grid);
        for (sample, value) in self.axis_iter(Axis(0)).zip(&values) {
            let _ = binned_statistic.add_sample(&sample, *value);
        }
        binned_statistic
    }

    private_impl! {}
}

/// Indicator for empty fields or values for binned statistic
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinContent<T>
where
    T: num_traits::Num,
{
    /// Empty bin
    Empty,
    /// Non-empty bin with some value `T`
    Value(T),
}

/// Implementation of negation operator for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// assert_eq!(-bin, BinContent::Value(-2.0));
/// ```
impl<T: num_traits::Num + core::ops::Neg + Neg<Output = T>> Neg for BinContent<T> {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            BinContent::Empty => Self::Empty,
            BinContent::Value(v) => Self::Value(-v),
        }
    }
}

/// Implementation of addition for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin + bin, BinContent::<f64>::Value(4.0));
/// assert_eq!(bin + empty_bin, BinContent::Value(2.0));
/// assert_eq!(empty_bin + bin, BinContent::Value(2.0));
/// assert_eq!(empty_bin + empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: num_traits::Num + Add<Output = T>> Add for BinContent<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        match (self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(v), BinContent::Empty) => Self::Value(v),
            (BinContent::Empty, BinContent::Value(w)) => Self::Value(w),
            (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v + w),
        }
    }
}

/// Implementation of addition assignment  for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin += empty_bin;
/// assert_eq!(bin, BinContent::Value(2.0));
///
/// empty_bin += bin;
/// assert_eq!(empty_bin, BinContent::Value(2.0));
///
/// bin += bin;
/// assert_eq!(bin, BinContent::Value(4.0));
///
/// let mut empty_bin = BinContent::<f64>::Empty;
/// empty_bin += empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: num_traits::Num + Copy> AddAssign for BinContent<T> {
    fn add_assign(&mut self, other: Self) {
        *self = match (&self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(v), BinContent::Empty) => Self::Value(*v),
            (BinContent::Empty, BinContent::Value(w)) => Self::Value(w),
            (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v + *w),
        };
    }
}

/// Implementation of substraction for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin - bin, BinContent::Value(0.0));
/// assert_eq!(bin - empty_bin, BinContent::Value(2.0));
/// assert_eq!(empty_bin - bin, BinContent::Value(-2.0));
/// assert_eq!(empty_bin - empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: num_traits::Num + Sub<Output = T>> Sub for BinContent<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        match (self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(v), BinContent::Empty) => Self::Value(v),
            (BinContent::Empty, BinContent::Value(w)) => {
                Self::Value(num_traits::identities::zero::<T>() - w)
            }
            (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v - w),
        }
    }
}

/// Implementation of substraction assignment  for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin += empty_bin;
/// assert_eq!(bin, BinContent::Value(2.0));
///
/// empty_bin += bin;
/// assert_eq!(empty_bin, BinContent::Value(2.0));
///
/// bin += bin;
/// assert_eq!(bin, BinContent::Value(4.0));
///
/// let mut empty_bin = BinContent::<f64>::Empty;
/// empty_bin += empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: num_traits::Num + Copy> SubAssign for BinContent<T> {
    fn sub_assign(&mut self, other: Self) {
        *self = match (&self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(v), BinContent::Empty) => Self::Value(*v),
            (BinContent::Empty, BinContent::Value(w)) => {
                Self::Value(num_traits::identities::zero::<T>() - w)
            }
            (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v - *w),
        };
    }
}

/// Implementation of multiplication for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin * bin, BinContent::Value(4.0));
/// assert_eq!(bin * empty_bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin * bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin * empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: num_traits::Num + Mul<Output = T>> Mul for BinContent<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        match (self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(_), BinContent::Empty) => Self::Empty,
            (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
            (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v * w),
        }
    }
}

/// Implementation of multiplication assignment for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin *= bin;
/// assert_eq!(bin, BinContent::Value(4.0));
///
/// bin *= empty_bin;
/// assert_eq!(bin, BinContent::<f64>::Empty);
///
/// let mut bin = BinContent::Value(2.0);
/// empty_bin *= bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
///
/// empty_bin *= empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: num_traits::Num + Copy> MulAssign for BinContent<T> {
    fn mul_assign(&mut self, other: Self) {
        *self = match (&self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(_), BinContent::Empty) => Self::Empty,
            (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
            (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v * *w),
        }
    }
}

/// Implementation of division for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin / bin, BinContent::Value(1.0));
/// assert_eq!(bin / empty_bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin / bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin / empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: num_traits::Num + Div<Output = T>> Div for BinContent<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        match (self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(_), BinContent::Empty) => Self::Empty,
            (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
            (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v / w),
        }
    }
}

/// Implementation of division assignment for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin /= bin;
/// assert_eq!(bin, BinContent::Value(1.0));
///
/// bin /= empty_bin;
/// assert_eq!(bin, BinContent::<f64>::Empty);
///
/// let mut bin = BinContent::Value(2.0);
/// empty_bin /= bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
///
/// empty_bin /= empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: num_traits::Num + Copy> DivAssign for BinContent<T> {
    fn div_assign(&mut self, other: Self) {
        *self = match (&self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(_), BinContent::Empty) => Self::Empty,
            (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
            (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v / *w),
        }
    }
}

/// Implementation of remainder for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(3.0);
/// let den = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin % den, BinContent::Value(1.0));
/// assert_eq!(bin % empty_bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin % bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin % empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: num_traits::Num + Div<Output = T>> Rem for BinContent<T> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        match (self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(_), BinContent::Empty) => Self::Empty,
            (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
            (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v % w),
        }
    }
}

/// Implementation of remainder assignment for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(3.0);
/// let mut den = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin %= den;
/// assert_eq!(bin, BinContent::Value(1.0));
///
/// bin %= empty_bin;
/// assert_eq!(bin, BinContent::<f64>::Empty);
///
/// let mut bin = BinContent::Value(3.0);
/// empty_bin %= bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
///
/// empty_bin %= empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
impl<T: num_traits::Num + Copy> RemAssign for BinContent<T> {
    fn rem_assign(&mut self, other: Self) {
        *self = match (&self, other) {
            (BinContent::Empty, BinContent::Empty) => Self::Empty,
            (BinContent::Value(_), BinContent::Empty) => Self::Empty,
            (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
            (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v % *w),
        }
    }
}

/// Implementation of zero-element (empty) for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
/// use num_traits::identities::Zero;
///
/// let bin = BinContent::zero();
/// assert_eq!(bin, BinContent::<f64>::Empty);
/// ```
impl<T: num_traits::Num> Zero for BinContent<T> {
    fn zero() -> Self {
        Self::Empty
    }
    fn is_zero(&self) -> bool {
        *self == Self::Empty
    }
}

/// Implementation of one-element (empty) for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
/// use num_traits::identities::One;
///
/// let bin = BinContent::one();
/// assert_eq!(bin, BinContent::Value(1.0));
/// ```
impl<T: num_traits::Num + One> One for BinContent<T> {
    fn one() -> Self {
        Self::Value(num_traits::identities::one())
    }
    fn is_one(&self) -> bool {
        *self == Self::Value(num_traits::identities::one())
    }
}
