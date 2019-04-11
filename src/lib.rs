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

pub use crate::correlation::CorrelationExt;
pub use crate::entropy::EntropyExt;
pub use crate::histogram::HistogramExt;
pub use crate::maybe_nan::{MaybeNan, MaybeNanExt};
pub use crate::quantile::{interpolate, Quantile1dExt, QuantileExt};
pub use crate::sort::Sort1dExt;
pub use crate::summary_statistics::SummaryStatisticsExt;

mod correlation;
mod entropy;
pub mod errors;
pub mod histogram;
mod maybe_nan;
mod quantile;
mod sort;
mod summary_statistics;
