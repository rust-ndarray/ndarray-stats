use std::collections::HashMap;
use std::hash::Hash;
use std::fmt;
mod bins;
use self::bins::{Bin1d, BinNd, BinsNd, Bins1d};
use ndarray::prelude::*;
use ndarray::Data;

type HistogramNd<T> = HashMap<BinNd<T>, usize>;

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
    fn histogram<B>(&self, bins: BinsNd<A>) -> HistogramNd<A>;
}

impl<A, S> HistogramNdExt<A, S> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    A: Hash + Eq + fmt::Debug + Clone + PartialOrd,
{
    fn histogram<B>(&self, bins: BinsNd<A>) -> HistogramNd<A>
    {
        let mut histogram = HashMap::new();
        for point in self.axis_iter(Axis(0)) {
            let bin = bins.find(point);
            if let Some(b) = bin {
                histogram.insert(b, 1);
            };
        }
        histogram
    }
}

type Histogram1d<T> = HashMap<Bin1d<T>, usize>;

pub trait Histogram1dExt<A, S>
where
    S: Data<Elem = A>,
    A: Hash + Eq + fmt::Debug + Clone,
{
    fn histogram<B>(&self, bins: Bins1d<A>) -> Histogram1d<A>;
}

impl<A, S> Histogram1dExt<A, S> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
    A: Hash + Eq + fmt::Debug + Clone + PartialOrd,
{
    fn histogram<B>(&self, bins: Bins1d<A>) -> Histogram1d<A>
    {
        let mut histogram = HashMap::new();
        for point in self.iter() {
            let bin = bins.find(point);
            if let Some(b) = bin {
                histogram.insert(b, 1);
            };
        }
        histogram
    }
}