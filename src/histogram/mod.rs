//! Histogram functionalities.
pub use self::histograms::{Histogram, HistogramExt};
pub use self::bins::{Edges, Bins};
pub use self::grid::{Grid, GridBuilder};
pub use self::errors::BinNotFound;

mod histograms;
mod bins;
pub mod builders;
mod grid;
mod errors;
