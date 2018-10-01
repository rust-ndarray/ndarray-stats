mod bins {
    use ndarray::prelude::*;
    use ndarray::Data;
    use std::hash::Hash;

    /// `Bins` is a collection of non-overlapping 
    /// intervals (`Bin1d`) in a 1-dimensional space.
    pub struct Bins1d<T: Hash + Eq> {
        bins: Vec<Bin1d<T>>,
    }
    /// `Bins` is a collection of non-overlapping 
    /// sub-regions (`BinNd`) in a `n`-dimensional space.
    pub struct BinsNd<T: Hash + Eq> {
        bins: Vec<BinNd<T>>,
    }

    impl<T> Bins1d<T> 
    where 
        T: Hash + Eq 
    {
        /// Given a point `P`, it returns an `Option`:
        /// - `Some(B)`, if `P` belongs to the `Bin` `B`;
        /// - `None`, if `P` does not belong to any `Bin` in `Bins`.
        /// 
        /// **Panics** if `P.ndim()` is different from `Bins.ndim()`. 
        pub fn find<S, D>(&self, _point: ArrayBase<S, D>) -> Option<Bin1d<T>>
        where
            S: Data<Elem = T>,
            D: Dimension,
        {
            unimplemented!()
        }
    }

    impl<T> BinsNd<T> 
    where 
        T: Hash + Eq 
    {
        /// Return `n`, the number of dimensions.
        fn ndim(&self) -> usize {
            unimplemented!() 
        }

        /// Given a point `P`, it returns an `Option`:
        /// - `Some(B)`, if `P` belongs to the `Bin` `B`;
        /// - `None`, if `P` does not belong to any `Bin` in `Bins`.
        /// 
        /// **Panics** if `P.ndim()` is different from `Bins.ndim()`. 
        pub fn find<S, D>(&self, _point: ArrayBase<S, D>) -> Option<BinNd<T>>
        where
            S: Data<Elem = T>,
            D: Dimension,
        {
            unimplemented!()
        }
    }

    #[derive(Hash, PartialEq, Eq)]
    pub struct Bin1d<T: Hash + Eq> {
        left: T,
        right: T,
    }

    #[derive(Hash, PartialEq, Eq)]
    pub struct BinNd<T: Hash + Eq> {
        projections: Vec<Bin1d<T>>,
    }
}

use std::collections::HashMap;
use std::hash::Hash;
use self::bins::{Bin1d, BinNd, BinsNd, Bins1d};
use ndarray::prelude::*;
use ndarray::Data;

type HistogramNd<T> = HashMap<BinNd<T>, usize>;

pub trait HistogramNdExt<A, S>
where
    S: Data<Elem = A>,
    A: Hash + Eq,
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
    A: Hash + Eq,
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
    A: Hash + Eq,
{
    fn histogram<B>(&self, bins: Bins1d<A>) -> Histogram1d<A>;
}

impl<A, S> Histogram1dExt<A, S> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
    A: Hash + Eq,
{
    fn histogram<B>(&self, bins: Bins1d<A>) -> Histogram1d<A>
    {
        unimplemented!()
    }
}