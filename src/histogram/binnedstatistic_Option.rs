/// 1. Bei Hinzufügen von Sample + Value immer alle Statistiken updaten (online)
/// 2. Oder: nur beim Bauen der Struktur (new) angeben welche Statistik bestimmt werden soll (match innerhalb add_sample)
/// 3. Update Documentation!!!
use super::errors::BinNotFound;
use super::grid::Grid;
use ndarray::prelude::*;
use ndarray::Data;
use std::cmp::Ordering;
// use std::ops::{Add, Mul};

// /// Staistics used for each bin.
// pub enum Statistic {
//     /// Compute the mean of values for points within each bin. \todo{How to represent no values in that bin?}
//     Mean,
//     /// Compute the median of values for points within each bin. \todo{How to represent no values in that bin?}
//     Median,
//     /// Compute the count of points within each bin. This is identical to an unweighted histogram.
//     Count,
//     /// Compute the sum of values for points within each bin (weighted histogram).
//     Sum,
//     /// Compute the standard deviation within each bin. \todo{How to represent no values in that bin?}
//     Std,
//     /// Compute the minimum of values for points within each bin. \todo{How to represent no values in that bin?}
//     Min,
//     /// Compute the maximum of values for point within each bin. \todo{How to represent no values in that bin?}
//     Max,
// }

/// Binned statistic data structure.
pub struct BinnedStatistic<A: Ord, T: num_traits::Num> {
    counts: ArrayD<Option<usize>>,
    sum: ArrayD<Option<T>>,
    m1: ArrayD<T>,
    m2: ArrayD<T>,
    mean: ArrayD<Option<T>>,
    variance: ArrayD<Option<T>>,
    standard_deviation: ArrayD<Option<T>>,
    min: ArrayD<Option<T>>,
    max: ArrayD<Option<T>>,
    grid: Grid<A>,
    ddof: usize,
}

// #[derive(Debug)]
// enum BinContent<T> {
//     Empty,
//     Value(T),
// }

// impl<T: Add<Output = T>> Add for BinContent<T> {
//     type Output = Self;

//     fn add(self, other: Self) -> Self {
//         match (self, other) {
//             (BinContent::Empty, BinContent::Empty) => Self::Empty,
//             (BinContent::Value(v), BinContent::Empty) => Self::Value(v),
//             (BinContent::Empty, BinContent::Value(v)) => Self::Value(v),
//             (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v + w),
//         }
//     }
// }

// impl<T: Mul<Output = T>> Mul<T> for BinContent<T> {
//     type Output = Self;

//     fn mul(self, rhs: T) -> Self {
//         match (self, rhs) {
//             (BinContent::Empty, _) => Self::Empty,
//             (BinContent::Value(v), rhs) => Self::Value(v * rhs),
//         }
//     }
// }

impl<A, T> BinnedStatistic<A, T>
where
    A: Ord,
    T: num_traits::Float + num_traits::FromPrimitive,
{
    /// Returns a new instance of BinnedStatistic given a [`Grid`].
    ///
    /// [`Grid`]: struct.Grid.html
    pub fn new(grid: Grid<A>, ddof: Option<usize>) -> Self {
        let counts = ArrayD::from_elem(grid.shape(), Option::None);
        let sum = ArrayD::from_elem(grid.shape(), Option::None);
        let m1 = ArrayD::zeros(grid.shape());
        let m2 = ArrayD::zeros(grid.shape());
        let mean = ArrayD::from_elem(grid.shape(), Option::None);
        let variance = ArrayD::from_elem(grid.shape(), Option::None);
        let standard_deviation = ArrayD::from_elem(grid.shape(), Option::None);
        let min = ArrayD::from_elem(grid.shape(), Option::None);
        let max = ArrayD::from_elem(grid.shape(), Option::None);
        BinnedStatistic {
            counts,
            sum,
            m1,
            m2,
            mean,
            variance,
            standard_deviation,
            min,
            max,
            grid,
            ddof: ddof.unwrap_or(0usize),
        }
    }

    /// Adds a single sample to the binned statistic.
    ///
    /// **Panics** if dimensions do not match: `self.ndim() != sample.len()`.
    ///
    /// # Example:
    /// ```
    /// use ndarray::array;
    /// use ndarray_stats::histogram::{Edges, Bins, BinnedStatistic, Grid, Statistic};
    /// use noisy_float::types::n64;
    ///
    /// let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    /// let bins = Bins::new(edges);
    /// let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    /// let mut binned_statistic = BinnedStatistic::new(square_grid, None);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, 1.0)?;
    /// binned_statistic.add_sample(&sample, 2.0)?;
    ///
    /// let binned_statistic_matrix = binned_statistic.sum();
    /// let expected = array![
    ///     [0.0, 0.0],
    ///     [0.0, 3.0],
    /// ];
    /// assert_eq!(binned_statistic_matrix, expected.into_dyn());
    /// # Ok::<(), Box<std::error::Error>>(())
    /// ```
    pub fn add_sample<S>(&mut self, sample: &ArrayBase<S, Ix1>, value: T) -> Result<(), BinNotFound>
    where
        S: Data<Elem = A>,
        T: Ord + num_traits::Float + num_traits::FromPrimitive,
    {
        match self.grid.index_of(sample) {
            Some(bin_index) => {
                // Updating storing `n` before updating `counts`
                // let n1 = match self.counts[&*bin_index] {
                //     None => 0i32,
                //     Some(x) => i32::try_from(x).unwrap(),
                // };
                let id = &*bin_index;
                // let n1 = i32::try_from(self.counts[id].unwrap_or(0)).unwrap();
                let n1 = self.counts[id].unwrap_or(0usize) as i32;
                let n = n1 + 1i32;
                self.counts[id] = match self.counts[id] {
                    None => Some(1),
                    Some(x) => Some(x + 1usize),
                };
                // Help variables
                let delta = value - self.m1[id];
                let delta_n = delta / T::from_i32(n).unwrap();
                let term1 = delta * delta_n * T::from_i32(n1).unwrap();

                // Intermediate variables for statistical moments
                self.m1[id] = self.m1[id] + delta_n;
                self.m2[id] = self.m2[id] + term1;

                // Mean
                self.mean[id] = Some(self.m1[id]);

                // Variance
                let dof = n - (self.ddof as i32);
                self.variance[id] = match dof.cmp(&0) {
                    Ordering::Less => panic!(
                        "`ddof` needs to be strictly smaller than the \
                         number of observations provided!  for each \
                         random variable! There are {} degrees of freedom left",
                        dof
                    ),
                    Ordering::Equal => Some(num_traits::identities::zero()),
                    Ordering::Greater => Some(self.m2[id] / T::from_i32(dof).unwrap()),
                };

                // Standard deviation (only enters when Some)
                self.standard_deviation[id] = if let Some(var) = self.variance[id] {
                    if var >= num_traits::identities::zero() {
                        Some(var.sqrt())
                    } else {
                        panic!(
                            "`variance` is negative, cannot take \
                             square root of negative number for \
                             `standard_deviation`!"
                        )
                    }
                } else {
                    None
                };

                // Sum
                self.sum[id] = match self.sum[id] {
                    None => Some(value),
                    Some(x) => Some(x + value),
                };

                // Max
                self.max[id] = match self.max[id] {
                    None => Some(value),
                    Some(x) => Some(num_traits::float::Float::max(value, x)),
                };

                // Min
                self.min[id] = match self.min[id] {
                    None => Some(value),
                    Some(x) => Some(num_traits::float::Float::min(value, x)),
                };
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
    pub fn sum(&self) -> ArrayViewD<'_, Option<T>> {
        self.sum.view()
    }

    /// Borrows a view on the binned statistic `counts` matrix (equivalent to histogram).
    pub fn counts(&self) -> ArrayViewD<'_, Option<usize>> {
        self.counts.view()
    }

    /// Borrows a view on the binned statistic `mean` matrix.
    pub fn mean(&self) -> ArrayViewD<'_, Option<T>> {
        self.mean.view()
    }

    /// Borrows a view on the binned statistic `variance` matrix.
    pub fn variance(&self) -> ArrayViewD<'_, Option<T>> {
        self.variance.view()
    }

    ///
    pub fn variance_2(&self) -> ArrayD<Option<T>> {
        let variance2 = ArrayD::from_elem(self.grid.shape(), Option::None);
        variance2
    }

    ///
    pub fn variance_3(&self) -> ArrayD<Option<T>> {
        self.variance.to_owned()
    }

    /// Borrows a view on the binned statistic `standard deviation` matrix.
    pub fn standard_deviation(&self) -> ArrayViewD<'_, Option<T>> {
        self.standard_deviation.view()
    }

    /// Borrows a view on the binned statistic max` matrix.
    pub fn max(&self) -> ArrayViewD<'_, Option<T>> {
        self.max.view()
    }

    /// Borrows a view on the binned statistic `min` matrix.
    pub fn min(&self) -> ArrayViewD<'_, Option<T>> {
        self.min.view()
    }

    /// Borrows an immutable reference to the binned statistic grid.
    pub fn grid(&self) -> &Grid<A> {
        &self.grid
    }
}

// impl<A: std::cmp::Ord, T: num_traits::Num + Add<Output = T>> Add for BinnedStatistic<A, T> {
//     type Output = Self;

//     fn add(self, other: Self) -> Self {
//         Self {
//             counts: self.counts.unwrap_or(num_traits::identities::zero()) + other.counts,
//             sum: self.sum + other.sum,
//         }
//     }
// }
