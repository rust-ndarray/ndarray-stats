//! The [`ndarray-stats`] crate exposes statistical routines for `ArrayBase`,
//! the *n*-dimensional array data structure provided by [`ndarray`].
//!
//! Currently available routines include:
//! - [`order statistics`] (minimum, maximum, quantiles, etc.);
//! - [`partitioning`];
//! - [`correlation analysis`] (covariance, pearson correlation);
//! - [`histogram computation`].
//!
//! Please feel free to contribute new functionality! A roadmap can be found [`here`].
//!
//! Our work is inspired by other existing statistical packages such as
//! [`NumPy`] (Python) and [`StatsBase.jl`] (Julia) - any contribution bringing us closer to
//! feature parity is more than welcome!
//!
//! [`ndarray-stats`]: https://github.com/jturner314/ndarray-stats/
//! [`ndarray`]: https://github.com/rust-ndarray/ndarray
//! [`order statistics`]: trait.QuantileExt.html
//! [`partitioning`]: trait.Sort1dExt.html
//! [`correlation analysis`]: trait.CorrelationExt.html
//! [`histogram computation`]: histogram/index.html
//! [`here`]: https://github.com/jturner314/ndarray-stats/issues/1
//! [`NumPy`]: https://docs.scipy.org/doc/numpy-1.14.1/reference/routines.statistics.html
//! [`StatsBase.jl`]: https://juliastats.github.io/StatsBase.jl/latest/


extern crate ndarray;
extern crate noisy_float;
extern crate num_traits;
extern crate rand;
extern crate itertools;

#[cfg(test)]
extern crate ndarray_rand;
#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
extern crate approx;

pub use maybe_nan::{MaybeNan, MaybeNanExt};
pub use quantile::{interpolate, QuantileExt, Quantile1dExt};
pub use sort::Sort1dExt;
pub use correlation::CorrelationExt;
pub use histogram::HistogramExt;
pub use summary_statistics::SummaryStatisticsExt;
pub use entropy::EntropyExt;

mod maybe_nan;
mod quantile;
mod sort;
mod correlation;
mod entropy;
mod summary_statistics;
pub mod errors;
pub mod histogram;
