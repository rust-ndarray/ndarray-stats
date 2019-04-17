//! The [`ndarray-stats`] crate exposes statistical routines for `ArrayBase`,
//! the *n*-dimensional array data structure provided by [`ndarray`].
//!
//! Currently available routines include:
//! - [order statistics] (minimum, maximum, median, quantiles, etc.);
//! - [summary statistics] (mean, skewness, kurtosis, central moments, etc.)
//! - [partitioning];
//! - [correlation analysis] (covariance, pearson correlation);
//! - [measures from information theory] (entropy, KL divergence, etc.);
//! - [histogram computation].
//!
//! Please feel free to contribute new functionality! A roadmap can be found [here].
//!
//! Our work is inspired by other existing statistical packages such as
//! [`NumPy`] (Python) and [`StatsBase.jl`] (Julia) - any contribution bringing us closer to
//! feature parity is more than welcome!
//!
//! [`ndarray-stats`]: https://github.com/rust-ndarray/ndarray-stats/
//! [`ndarray`]: https://github.com/rust-ndarray/ndarray
//! [order statistics]: trait.QuantileExt.html
//! [partitioning]: trait.Sort1dExt.html
//! [summary statistics]: trait.SummaryStatisticsExt.html
//! [correlation analysis]: trait.CorrelationExt.html
//! [measures from information theory]: trait.EntropyExt.html
//! [histogram computation]: histogram/index.html
//! [here]: https://github.com/rust-ndarray/ndarray-stats/issues/1
//! [`NumPy`]: https://docs.scipy.org/doc/numpy-1.14.1/reference/routines.statistics.html
//! [`StatsBase.jl`]: https://juliastats.github.io/StatsBase.jl/latest/

pub use crate::correlation::CorrelationExt;
pub use crate::deviation::DeviationExt;
pub use crate::entropy::EntropyExt;
pub use crate::histogram::HistogramExt;
pub use crate::maybe_nan::{MaybeNan, MaybeNanExt};
pub use crate::quantile::{interpolate, Quantile1dExt, QuantileExt};
pub use crate::sort::Sort1dExt;
pub use crate::summary_statistics::SummaryStatisticsExt;

#[macro_use]
mod private {
    /// This is a public type in a private module, so it can be included in
    /// public APIs, but other crates can't access it.
    pub struct PrivateMarker;

    /// Defines an associated function for a trait that is impossible for other
    /// crates to implement. This makes it possible to add new associated
    /// types/functions/consts/etc. to the trait without breaking changes.
    macro_rules! private_decl {
        () => {
            /// This method makes this trait impossible to implement outside of
            /// `ndarray-stats` so that we can freely add new methods, etc., to
            /// this trait without breaking changes.
            ///
            /// We don't anticipate any other crates needing to implement this
            /// trait, but if you do have such a use-case, please let us know.
            ///
            /// **Warning** This method is not considered part of the public
            /// API, and client code should not rely on it being present. It
            /// may be removed in a non-breaking release.
            fn __private__(&self, _: crate::private::PrivateMarker);
        };
    }

    /// Implements the associated function defined by `private_decl!`.
    macro_rules! private_impl {
        () => {
            fn __private__(&self, _: crate::private::PrivateMarker) {}
        };
    }
}

mod correlation;
mod deviation;
mod entropy;
pub mod errors;
pub mod histogram;
mod maybe_nan;
mod quantile;
mod sort;
mod summary_statistics;
