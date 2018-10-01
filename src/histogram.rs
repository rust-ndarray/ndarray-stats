mod bins {
    use ndarray::prelude::*;
    use ndarray::Data;

    struct Bins1d {
        bins: Vec<Bin1d>,
    }
    struct BinsNd {
        bins: Vec<BinNd>,
    }

    /// `Bins` is a collection of non-overlapping
    /// sub-regions (`Bin`) in a `n` dimensional space.
    pub trait Bins {
        /// Return `n`, the number of dimensions.
        fn ndim(&self) -> usize;

        /// Given a point `P`, it returns an `Option`:
        /// - `Some(B)`, if `P` belongs to the `Bin` `B`;
        /// - `None`, if `P` does not belong to any `Bin` in `Bins`.
        /// 
        /// **Panics** if `P.ndim()` is different from `Bins.ndim()`. 
        fn find<A, S, D, B>(&self, point: ArrayBase<S, D>) -> Option<B>
        where
            S: Data<Elem = A>,
            D: Dimension,
            B: Bin;
    }

    impl Bins for Bins1d {
        fn ndim(&self) -> usize {
            1
        }

        fn find<A, S, D, B>(&self, _point: ArrayBase<S, D>) -> Option<B>
        where
            S: Data<Elem = A>,
            D: Dimension,
            B: Bin,
        {
            unimplemented!()
        }
    }

    impl Bins for BinsNd {
        fn ndim(&self) -> usize {
            unimplemented!() 
        }

        fn find<A, S, D, B>(&self, _point: ArrayBase<S, D>) -> Option<B>
        where
            S: Data<Elem = A>,
            D: Dimension,
            B: Bin,
        {
            unimplemented!()
        }
    }

    pub trait Bin {

    }
    struct Bin1d {}
    struct BinNd {}
    impl Bin for Bin1d {}
    impl Bin for BinNd {}
}

pub trait HistogramExt {

}