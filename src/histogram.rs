mod bins {
    use ndarray::prelude::*;
    use ndarray::Data;

    struct Bins1d<T> {
        bins: Vec<Bin1d<T>>,
    }
    struct BinsNd<T> {
        bins: Vec<BinNd<T>>,
    }

    /// `Bins` is a collection of non-overlapping
    /// sub-regions (`Bin`) in a `n` dimensional space.
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

pub trait HistogramExt {

}