pub use self::histograms::{HistogramCounts, HistogramExt};
pub use self::bins::{Edges, Bins};
pub use self::errors::BinNotFound;

mod histograms;
mod bins;
mod errors;
