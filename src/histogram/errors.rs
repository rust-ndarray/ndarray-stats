use std::error;
use std::fmt;

/// Error to denote that no bin has been found for a certain observation.
#[derive(Debug, Clone)]
pub struct BinNotFound;

impl fmt::Display for BinNotFound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "No bin has been found.")
    }
}

impl error::Error for BinNotFound {
    fn description(&self) -> &str {
        "No bin has been found."
    }

    fn cause(&self) -> Option<&error::Error> {
        // Generic error, underlying cause isn't tracked.
        None
    }
}
