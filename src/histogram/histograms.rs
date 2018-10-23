use ndarray::prelude::*;
use ndarray::Data;
use super::bins::Bins;
use super::errors::BinNotFound;

pub struct HistogramCounts<A: Ord> {
    counts: ArrayD<usize>,
    bins: Vec<Bins<A>>,
}

impl<A: Ord> HistogramCounts<A> {
    /// Return a new instance of HistogramCounts given
    /// a vector of [`Bins`].
    ///
    /// The `i`-th element in `Vec<Bins<A>>` represents the 1-dimensional
    /// projection of the bin grid on the `i`-th axis.
    ///
    /// [`Bins`]: struct.Bins.html
    pub fn new(bins: Vec<Bins<A>>) -> Self {
        let ndim = bins.len();
        let counts = ArrayD::zeros(
            bins.iter().map(|e| e.len()
            ).collect::<Vec<_>>());
        HistogramCounts { counts, bins }
    }

    /// Add a single observation to the histogram.
    ///
    /// **Panics** if dimensions do not match: `self.ndim() != observation.len()`.
    pub fn add_observation(&mut self, observation: ArrayView1<A>) -> Result<(), BinNotFound> {
        assert_eq!(
            self.ndim,
            observation.len(),
            "Dimensions do not match: observation has {0} dimensions, \
             while the histogram has {1}.", observation.len(), self.ndim
        );
        let bin = observation
            .iter()
            .zip(&self.bins)
            .map(|(v, e)| e.index(v).ok_or(BinNotFound))
            .collect::<Result<Vec<_>, _>>()?;
        self.counts[IxDyn(&bin)] += 1;
        Ok(())
    }

    /// Returns the number of dimensions of the space the histogram is covering.
    pub fn ndim(&self) -> usize {
        debug_assert_eq!(self.counts.ndim(), self.bins.len());
        self.counts.len()
    }
}

/// Histogram methods.
pub trait HistogramExt<A, S>
    where
        S: Data<Elem = A>,
{
    /// Return the [histogram](https://en.wikipedia.org/wiki/Histogram)
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
    /// **Panics** if `d` is different from `bins.len()`.
    fn histogram(&self, bins: Vec<Bins<A>>) -> HistogramCounts<A>
        where
            A: Ord;
}

impl<A, S> HistogramExt<A, S> for ArrayBase<S, Ix2>
    where
        S: Data<Elem = A>,
        A: Ord,
{
    fn histogram(&self, bins: Vec<Bins<A>>) -> HistogramCounts<A>
    {
        let mut histogram = HistogramCounts::new(bins);
        for point in self.axis_iter(Axis(0)) {
            histogram.add_observation(point);
        }
        histogram
    }
}