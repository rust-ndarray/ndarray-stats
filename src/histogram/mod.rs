//! Histogram functionalities.
pub use self::histograms::{Histogram, HistogramExt};
pub use self::bins::{Edges, Bins};
pub use self::grid::{Grid, GridBuilder};

mod histograms;
mod bins;
pub mod strategies;
mod grid;
pub mod errors;
