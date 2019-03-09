//! Histogram functionalities.
pub use self::bins::{Bins, Edges};
pub use self::grid::{Grid, GridBuilder};
pub use self::histograms::{Histogram, HistogramExt};

mod bins;
pub mod errors;
mod grid;
mod histograms;
pub mod strategies;
