/// 1. Bei Hinzufügen von Sample + Value immer alle Statistiken updaten (online)
/// 2. Oder: nur beim Bauen der Struktur (new) angeben welche Statistik bestimmt werden soll (match innerhalb add_sample)
/// 3. Update Documentation!!!
use super::errors::BinNotFound;
use super::grid::Grid;
use ndarray::prelude::*;
use ndarray::Data;
use std::ops::AddAssign;

/// Binned statistic data structure.
pub struct BinnedStatistic<A: Ord, T> {
    values: ArrayD<T>,
    counts: ArrayD<i32>,
    m1: ArrayD<T>,
    m2: ArrayD<T>,
    m3: ArrayD<T>,
    m4: ArrayD<T>,
    grid: Grid<A>,
}

/// Staistics used for each bin.
pub enum Statistic {
    /// Compute the mean of values for points within each bin. \todo{How to represent no values in that bin?}
    Mean,
    /// Compute the median of values for points within each bin. \todo{How to represent no values in that bin?}
    Median,
    /// Compute the count of points within each bin. This is identical to an unweighted histogram.
    Count,
    /// Compute the sum of values for points within each bin (weighted histogram).
    Sum,
    /// Compute the standard deviation within each bin. \todo{How to represent no values in that bin?}
    Std,
    /// Compute the minimum of values for points within each bin. \todo{How to represent no values in that bin?}
    Min,
    /// Compute the maximum of values for point within each bin. \todo{How to represent no values in that bin?}
    Max,
}

impl<A, T> BinnedStatistic<A, T>
where
    A: Ord,
    T: Clone
        + AddAssign
        + num_traits::identities::Zero
        + std::ops::Add
        + std::ops::Sub
        + std::ops::Div
        + std::ops::Mul
        + std::convert::AsRef<i32>
        + std::convert::From<i32>,
{
    /// Returns a new instance of BinnedStatistic given a [`Grid`].
    ///
    /// [`Grid`]: struct.Grid.html
    pub fn new(grid: Grid<A>) -> Self {
        let values = ArrayD::zeros(grid.shape());
        let counts = ArrayD::zeros(grid.shape());
        let m1 = ArrayD::zeros(grid.shape());
        let m2 = ArrayD::zeros(grid.shape());
        let m3 = ArrayD::zeros(grid.shape());
        let m4 = ArrayD::zeros(grid.shape());
        BinnedStatistic {
            values,
            counts,
            m1,
            m2,
            m3,
            m4,
            grid,
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
    /// let mut binned_statistic = BinnedStatistic::new(square_grid, Statistic::Sum);
    ///
    /// let sample = array![n64(0.5), n64(0.6)];
    ///
    /// binned_statistic.add_sample(&sample, 1.0)?;
    /// binned_statistic.add_sample(&sample, 2.0)?;
    ///
    /// let binned_statistic_matrix = binned_statistic.values();
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
    {
        match self.grid.index_of(sample) {
            Some(bin_index) => {
                // Updating storing `n` before updating `counts`
                let n1 = self.counts[&*bin_index];
                self.counts[&*bin_index] += 1;
                let n = self.counts[&*bin_index];
                // Help variables
                let delta = value - self.m1[&*bin_index];
                let delta_n = delta / n;
                let delta_n2 = delta_n * delta_n;
                let term1 = delta * delta_n * n1;

                // Intermediate variables for statistical moments
                self.m1 = delta_n;
                self.m4 = term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2
                    - 4.0 * delta_n * self.m3;
                self.m3 = term1 * delta_n * (n1 - 2) - 3 * delta_n * self.m2;
                self.m2 = term1;

                self.values[&*bin_index] += value;
                Ok(())
            }
            None => Err(BinNotFound),
        }
    }

    /// Returns the number of dimensions of the space the binned statistic is covering.
    pub fn ndim(&self) -> usize {
        debug_assert_eq!(self.values.ndim(), self.grid.ndim());
        self.values.ndim()
    }

    /// Borrows a view on the binned statistic values matrix.
    pub fn values(&self) -> ArrayViewD<'_, T> {
        self.values.view()
    }

    /// Borrows a view on the binned statistic counts matrix (equivalent to histogram).
    pub fn counts(&self) -> ArrayViewD<'_, i32> {
        self.counts.view()
    }

    /// Borrows a view on the binned statistic `mean`.
    pub fn mean(&self) -> ArrayViewD<'_, T> {
        self.m1.view()
    }

    /// Borrows a view on the binned statistic `variance`.
    pub fn variance(&self) -> ArrayViewD<'_, T> {
        (self.m2 / (self.counts - 1.0)).view()
    }

    /// Borrows a view on the binned statistic `standard deviation`.
    pub fn standard_deviation(&self) -> ArrayViewD<'_, T> {
        (self.m2 / (self.counts - 1.0)).mapv(T::sqrt).view()
    }

    /// Borrows a view on the binned statistic `skewness`.
    pub fn skewness(&self) -> ArrayViewD<'_, T> {
        (self.counts.mapv(T::sqrt) * self.m3 / self.m2.mapv(|x| x.powf(1.5))).view()
    }

    /// Borrows a view on the binned statistic `kurtosis`.
    pub fn kurtosis(&self) -> ArrayViewD<'_, T> {
        (self.counts * self.m4 / (self.m2 * self.m2) - 3.0).view()
    }

    /// Borrows an immutable reference to the binned statistic grid.
    pub fn grid(&self) -> &Grid<A> {
        &self.grid
    }
}

// ArrayBase<S, Ix2>.histogram(grid)
// ArrayBase<S, Ix2>.histogram_stats(values, grid, statistic)
// bs = BinnedStatistic::new(...)
// bs = BinnedStatistic::with_samples(grid, samples, values, statistic) // eqv zu scipy
