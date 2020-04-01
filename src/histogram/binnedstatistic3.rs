/// 1. Bei Hinzufügen von Sample + Value immer alle Statistiken updaten (online)
/// 2. Oder: nur beim Bauen der Struktur (new) angeben welche Statistik bestimmt werden soll (match innerhalb add_sample)
/// 3. Update Documentation!!!
use super::errors::BinNotFound;
use super::grid::Grid;
use ndarray::prelude::{ArrayBase, ArrayD, ArrayViewD, Ix1};
// use ndarray::{Data, Zip};
// use std::cmp::Ordering;
use std::ops::Add;

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
pub struct BinnedStatistic2<A: Ord, T: num_traits::Num> {
    counts: ArrayD<usize>,
    sum: ArrayD<T>,
    // m1: ArrayD<T>,
    // m2: ArrayD<T>,
    // variance: ArrayD<T>,
    // standard_deviation: ArrayD<T>,
    // min: ArrayD<T>,
    // max: ArrayD<T>,
    grid: Grid<A>,
    // ddof: usize,
}

impl<A, T> BinnedStatistic2<A, T>
where
    A: Ord,
    T: num_traits::Float + num_traits::FromPrimitive,
{
    /// Returns a new instance of BinnedStatistic given a [`Grid`].
    ///
    /// [`Grid`]: struct.Grid.html
    pub fn new(grid: Grid<A>, ddof: Option<usize>) -> Self {
        let counts = ArrayD::from_elem(grid.shape(), 0usize);
        let sum = ArrayD::from_elem(grid.shape(), num_traits::identities::zero());
        // let m1 = ArrayD::zeros(grid.shape());
        // let m2 = ArrayD::zeros(grid.shape());
        // let variance = ArrayD::from_elem(grid.shape(), num_traits::identities::zero());
        // let standard_deviation = ArrayD::from_elem(grid.shape(), num_traits::identities::zero());
        // let min = ArrayD::from_elem(
        //     grid.shape(),
        //     num_traits::identities::one::<T>() / num_traits::identities::zero::<T>(),
        // );
        // let max = ArrayD::from_elem(
        //     grid.shape(),
        //     -num_traits::identities::one::<T>() / num_traits::identities::zero::<T>(),
        // );
        BinnedStatistic2 {
            counts,
            sum,
            // m1,
            // m2,
            // variance,
            // standard_deviation,
            // min,
            // max,
            grid,
            // ddof: ddof.unwrap_or(0usize),
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
                // // let n1 = i32::try_from(self.counts[id].unwrap_or(0)).unwrap();
                // let n1 = self.counts[id] as i32;
                // let n = n1 + 1i32;
                self.counts[id] += 1usize;
                // // Help variables
                // let delta = value - self.m1[id];
                // let delta_n = delta / T::from_i32(n).unwrap();
                // let term1 = delta * delta_n * T::from_i32(n1).unwrap();

                // // Intermediate variables for statistical moments
                // self.m1[id] = self.m1[id] + delta_n;
                // self.m2[id] = self.m2[id] + term1;

                // // Variance
                // let dof = n - (self.ddof as i32);
                // self.variance[id] = match dof.cmp(&0) {
                //     Ordering::Less => panic!(
                //         "`ddof` needs to be strictly smaller than the \
                //          number of observations provided!  for each \
                //          random variable! There are {} degrees of freedom left",
                //         dof
                //     ),
                //     Ordering::Equal => num_traits::identities::zero(),
                //     Ordering::Greater => self.m2[id] / T::from_i32(dof).unwrap(),
                // };

                // // Standard deviation (only enters when Some)
                // self.standard_deviation[id] = if self.variance[id] >= num_traits::identities::zero()
                // {
                //     self.variance[id].sqrt()
                // } else {
                //     panic!(
                //         "`variance` is negative, cannot take \
                //              square root of negative number for \
                //              `standard_deviation`!"
                //     )
                // };

                // Sum
                self.sum[id] = self.sum[id] + value;

                // // Max
                // self.max[id] = num_traits::float::Float::max(value, self.max[id]);

                // // Min
                // self.min[id] = num_traits::float::Float::min(value, self.min[id]);
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

    // /// Borrows a view on the binned statistic `mean` matrix.
    // pub fn mean(&self) -> ArrayViewD<'_, T> {
    //     self.m1.view()
    // }

    // /// Borrows a view on the binned statistic `variance` matrix.
    // pub fn variance(&self) -> ArrayViewD<'_, T> {
    //     self.variance.view()
    // }

    // /// Borrows a view on the binned statistic `standard deviation` matrix.
    // pub fn standard_deviation(&self) -> ArrayViewD<'_, T> {
    //     self.standard_deviation.view()
    // }

    // /// Borrows a view on the binned statistic max` matrix.
    // pub fn max(&self) -> ArrayViewD<'_, T> {
    //     self.max.view()
    // }

    // /// Borrows a view on the binned statistic `min` matrix.
    // pub fn min(&self) -> ArrayViewD<'_, T> {
    //     self.min.view()
    // }

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

// impl<
//         A: Ord,
//         T: std::fmt::Debug + num_traits::Float + num_traits::FromPrimitive + Add<Output = T>,
//     > Add for BinnedStatistic2<A, T>
// {
//     type Output = Self;

//     fn add(self, other: Self) -> Self {
//         if self.grid != other.grid {
//             panic!("`BinnedStatistics` can only be added for same `grid`!")
//         };
//         if self.ddof != other.ddof {
//             panic!("`BinnedStatistics` can only be added for same `ddof`!")
//         };

//         let n_self = self.counts.mapv(|x| T::from_i32(x as i32).unwrap());
//         let n_other = other.counts.mapv(|x| T::from_i32(x as i32).unwrap());
//         let n_combined = &n_self + &n_other;
//         let delta = &other.m1 - &self.m1;
//         let delta2 = &delta * &delta;
//         let n2 = (&self.counts * &other.counts).mapv(|x| T::from_i32(x as i32).unwrap());
//         let mut m1 = n_self * &self.m1 + n_other * &other.m1;
//         Zip::from(&mut m1).and(&n_combined).apply(|a, &b| {
//             *a = if b == num_traits::identities::zero() {
//                 num_traits::identities::zero()
//             } else {
//                 *a / b
//             }
//         });
//         //let mut m2 = &self.m2 + &other.m2 + delta2 * n2 / &n_combined;
//         let mut m2 = delta2 * n2;
//         Zip::from(&mut m2)
//             .and(&self.m2)
//             .and(&other.m2)
//             .and(&n_combined)
//             .apply(|a, &b, &c, &d| {
//                 *a = if d == num_traits::identities::zero() {
//                     num_traits::identities::zero()
//                 } else {
//                     *a / d + b + c
//                 }
//             });
//         let dof = n_combined.mapv(|x| x - T::from_i32(self.ddof as i32).unwrap());
//         let mut variance = m2.clone();
//         Zip::from(&mut variance).and(&dof).apply(|a, &b| {
//             *a = if b == num_traits::identities::zero() {
//                 num_traits::identities::zero()
//             } else {
//                 *a / b
//             }
//         });
//         let mut min = self.min.clone();
//         Zip::from(&mut min).and(&other.min).apply(|a, &b| {
//             *a = num_traits::float::Float::min(*a, b);
//         });
//         let mut max = self.min.clone();
//         Zip::from(&mut max).and(&other.max).apply(|a, &b| {
//             *a = num_traits::float::Float::max(*a, b);
//         });

//         BinnedStatistic2 {
//             counts: &self.counts + &other.counts,
//             sum: self.sum + other.sum,
//             m1: m1.to_owned(),
//             m2: m2.to_owned(),
//             variance: variance.to_owned(),
//             standard_deviation: variance.mapv(T::sqrt),
//             min: min.clone(),
//             max: self.max,
//             grid: self.grid,
//             ddof: self.ddof,
//         }
//     }
// }

// impl<T: Mul<Output = T>> Mul<T> for BinContent<T> {
//     type Output = Self;

//     fn mul(self, rhs: T) -> Self {
//         match (self, rhs) {
//             (BinContent::Empty, rhs) => Self::Empty,
//             (BinContent::Value(v), rhs) => Self::Value(v * rhs),
//         }
//     }
// }
