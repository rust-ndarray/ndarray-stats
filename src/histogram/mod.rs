mod bins;
pub use self::bins::{Bin1d, BinNd, BinsNd, Bins1d};

use std::collections::HashMap;
use std::hash::Hash;
use std::fmt;
use ndarray::prelude::*;
use ndarray::Data;

type HistogramNd<T> = HashMap<BinNd<T>, usize>;

/// Extension trait for ArrayBase providing methods to compute n-dimensional histograms.
pub trait HistogramNdExt<A, S>
where
    S: Data<Elem = A>,
    A: Hash + Eq + fmt::Debug + Clone,
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
    /// **Panics** if `d` is different from `bins.ndim()`.
    fn histogram(&self, bins: BinsNd<A>) -> HistogramNd<A>;
}

impl<A, S> HistogramNdExt<A, S> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    A: Hash + Eq + fmt::Debug + Clone + PartialOrd,
{
    fn histogram(&self, bins: BinsNd<A>) -> HistogramNd<A>
    {
        let mut histogram = HashMap::new();
        for point in self.axis_iter(Axis(0)) {
            let bin = bins.find(point.view());
            if let Some(b) = bin {
                let counter = histogram.entry(b).or_insert(0);
                *counter += 1;
            };
        }
        histogram
    }
}

type Histogram1d<T> = HashMap<Bin1d<T>, usize>;

/// Extension trait for one-dimensional ArrayBase providing methods to compute histograms.
pub trait Histogram1dExt<A, S>
where
    S: Data<Elem = A>,
    A: Hash + Eq + fmt::Debug + Clone,
{
    /// Return the [histogram](https://en.wikipedia.org/wiki/Histogram)
    /// for a 1-dimensional array of points `M`.
    fn histogram(&self, bins: Bins1d<A>) -> Histogram1d<A>;
}

impl<A, S> Histogram1dExt<A, S> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
    A: Hash + Eq + fmt::Debug + Clone + PartialOrd,
{
    fn histogram(&self, bins: Bins1d<A>) -> Histogram1d<A>
    {
        let mut histogram = HashMap::new();
        for point in self.iter() {
            let bin = bins.find(point);
            if let Some(b) = bin {
                let counter = histogram.entry(b).or_insert(0);
                *counter += 1;
            };
        }
        histogram
    }
}

#[cfg(test)]
mod histogram_nd_tests {
    use super::*;

    #[test]
    fn histogram() {
        let first_quadrant = BinNd::new(
            vec![Bin1d::RangeFrom(0..),
                 Bin1d::RangeFrom(0..)
            ]
        );
        let second_quadrant = BinNd::new(
            vec![Bin1d::RangeTo(..0),
                 Bin1d::RangeFrom(0..)
            ]
        );
        let bins = BinsNd::new(vec![first_quadrant.clone(),
                                    second_quadrant.clone()]);
        let points = array![
            [1, 1],
            [1, 2],
            [0, 1],
            [-1, 2],
            [-1, -1], // a point that doesn't belong to any bin in bins
        ];
        assert_eq!(points.shape(), &[5, 2]);
        let histogram = points.histogram(bins);

        let mut expected = HashMap::new();
        expected.insert(first_quadrant, 3);
        expected.insert(second_quadrant, 1);

        assert_eq!(expected, histogram);
    }

    #[test]
    #[should_panic]
    fn histogram_w_mismatched_dimensions() {
        let bin = BinNd::new(vec![Bin1d::RangeFrom(0..)]);
        let bins = BinsNd::new(vec![bin.clone()]);
        let points = array![
            [1, 1],
            [1, 2],
        ];
        assert_eq!(points.shape(), &[2, 2]);
        points.histogram(bins);
    }
}