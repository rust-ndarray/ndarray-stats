mod bins {
    use ndarray::prelude::*;
    use ndarray::Data;

    struct Bins1d<T> {
        bins: Vec<Bin1d<T>>,
    }
    struct BinsNd<T> {
        bins: Vec<BinNd<T>>,
    }

    /// `Bins` is a collection of non-overlapping /// sub-regions (`Bin`) in a `n` dimensional space.
    pub trait Bins<T> {
        /// Return `n`, the number of dimensions.
        fn ndim(&self) -> usize;

        /// Given a point `P`, it returns an `Option`:
        /// - `Some(B)`, if `P` belongs to the `Bin` `B`;
        /// - `None`, if `P` does not belong to any `Bin` in `Bins`.
        /// 
        /// **Panics** if `P.ndim()` is different from `Bins.ndim()`. 
        fn find<S, D, B>(&self, point: ArrayBase<S, D>) -> Option<B>
        where
            S: Data<Elem = T>,
            D: Dimension,
            B: Bin;
    }

    impl<T> Bins<T> for Bins1d<T> {
        fn ndim(&self) -> usize {
            1
        }

        fn find<S, D, B>(&self, _point: ArrayBase<S, D>) -> Option<B>
        where
            S: Data<Elem = T>,
            D: Dimension,
            B: Bin,
        {
            unimplemented!()
        }
    }

    impl<T> Bins<T> for BinsNd<T> {
        fn ndim(&self) -> usize {
            unimplemented!() 
        }

        fn find<S, D, B>(&self, _point: ArrayBase<S, D>) -> Option<B>
        where
            S: Data<Elem = T>,
            D: Dimension,
            B: Bin,
        {
            unimplemented!()
        }
    }

    pub trait Bin {

    }
    struct Bin1d<T> {
        left: T,
        right: T,
    }
    struct BinNd<T> {
        projections: Vec<Bin1d<T>>,
    }
    impl<T> Bin for Bin1d<T> {}
    impl<T> Bin for BinNd<T> {}
}

use std::collections::HashMap;
use self::bins::{Bin, Bins};
use ndarray::prelude::*;
use ndarray::Data;

type Histogram = HashMap<Box<Bin>, usize>;

pub trait HistogramNdExt<A, S>
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
    fn histogram<B>(&self, bins: B) -> Histogram
    where
        B: Bins<A>;
}

impl<A, S> HistogramNdExt<A, S> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
{
    fn histogram<B>(&self, bins: B) -> Histogram
    where
        B: Bins<A>,
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

pub trait Histogram1dExt<A, S>
where
    S: Data<Elem = A>,
{
    fn histogram<B>(&self, bins: B) -> Histogram
    where
        B: Bins<A>;
}

impl<A, S> Histogram1dExt<A, S> for ArrayBase<S, Ix1>
where
    S: Data<Elem = A>,
{
    fn histogram<B>(&self, _bins: B) -> Histogram
    where
        B: Bins<A>,
    {
        unimplemented!()
    }
}