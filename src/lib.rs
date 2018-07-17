#[macro_use(s)]
extern crate ndarray;
extern crate noisy_float;

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck;

pub use maybe_nan::MaybeNan;
pub use min_max::MinMaxExt;

mod maybe_nan;
mod min_max;
