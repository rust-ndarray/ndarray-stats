#[macro_use(azip, s)]
#[cfg_attr(test, macro_use(array))]
extern crate ndarray;
extern crate noisy_float;
extern crate num_traits;
extern crate rand;

#[cfg(test)]
extern crate ndarray_rand;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck;

pub use maybe_nan::{MaybeNan, MaybeNanExt};
pub use quantile::{interpolate, QuantileExt};
pub use sort::Sort1dExt;
pub use correlation::CorrelationExt;
pub use histogram::{Histogram1dExt, HistogramNdExt, bins};

mod maybe_nan;
mod quantile;
mod sort;
mod correlation;
mod histogram;