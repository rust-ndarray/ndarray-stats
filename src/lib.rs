//! The [`ndarray-stats`] crate exposes statistical routines for `ArrayBase`,
//! the *n*-dimensional array data structure provided by [`ndarray`].
//!
//! Currently available routines include:
//! - [order statistics] (minimum, maximum, quantiles, etc.);
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
//! [`ndarray-stats`]: https://github.com/jturner314/ndarray-stats/
//! [`ndarray`]: https://github.com/rust-ndarray/ndarray
//! [order statistics]: trait.QuantileExt.html
//! [partitioning]: trait.Sort1dExt.html
//! [correlation analysis]: trait.CorrelationExt.html
//! [measures from information theory]: trait.EntropyExt.html
//! [histogram computation]: histogram/index.html
//! [here]: https://github.com/jturner314/ndarray-stats/issues/1
//! [`NumPy`]: https://docs.scipy.org/doc/numpy-1.14.1/reference/routines.statistics.html
//! [`StatsBase.jl`]: https://juliastats.github.io/StatsBase.jl/latest/

extern crate indexmap;
extern crate itertools;
extern crate ndarray;
extern crate noisy_float;
extern crate num_integer;
extern crate num_traits;
extern crate rand;

#[cfg(test)]
extern crate approx;
#[cfg(test)]
extern crate ndarray_rand;
#[cfg(test)]
extern crate quickcheck;

pub use correlation::CorrelationExt;
pub use entropy::EntropyExt;
pub use histogram::HistogramExt;
pub use maybe_nan::{MaybeNan, MaybeNanExt};
pub use quantile::{interpolate, Quantile1dExt, QuantileExt};
pub use sort::Sort1dExt;
pub use summary_statistics::SummaryStatisticsExt;

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
            /// `ndarray-stats`.
            fn __private__(&self) -> crate::private::PrivateMarker;
        };
    }

    /// Implements the associated function defined by `private_decl!`.
    macro_rules! private_impl {
        () => {
            fn __private__(&self) -> crate::private::PrivateMarker {
                crate::private::PrivateMarker
            }
        };
    }
}

mod correlation;
mod entropy;
pub mod errors;
pub mod histogram;
mod maybe_nan;
mod quantile;
mod sort;
mod summary_statistics;
