use crate::errors::{EmptyInput, MinMaxError};
use std::error;
use std::fmt;

/// Error to denote that no bin has been found for a certain observation.
#[derive(Debug, Clone)]
pub struct BinNotFound;

impl fmt::Display for BinNotFound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "No bin has been found.")
    }
}

impl error::Error for BinNotFound {
    fn description(&self) -> &str {
        "No bin has been found."
    }
}

/// Error computing the set of histogram bins.
#[derive(Debug, Clone)]
pub enum BinsBuildError {
    /// The input array was empty.
    EmptyInput,
    /// The strategy for computing appropriate bins failed.
    Strategy,
    #[doc(hidden)]
    __NonExhaustive,
}

impl BinsBuildError {
    /// Returns whether `self` is the `EmptyInput` variant.
    pub fn is_empty_input(&self) -> bool {
        match self {
            BinsBuildError::EmptyInput => true,
            _ => false,
        }
    }

    /// Returns whether `self` is the `Strategy` variant.
    pub fn is_strategy(&self) -> bool {
        match self {
            BinsBuildError::Strategy => true,
            _ => false,
        }
    }
}

impl fmt::Display for BinsBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "The strategy failed to determine a non-zero bin width.")
    }
}

impl error::Error for BinsBuildError {
    fn description(&self) -> &str {
        "The strategy failed to determine a non-zero bin width."
    }
}

impl From<EmptyInput> for BinsBuildError {
    fn from(_: EmptyInput) -> Self {
        BinsBuildError::EmptyInput
    }
}

impl From<MinMaxError> for BinsBuildError {
    fn from(err: MinMaxError) -> BinsBuildError {
        match err {
            MinMaxError::EmptyInput => BinsBuildError::EmptyInput,
            MinMaxError::UndefinedOrder => BinsBuildError::Strategy,
        }
    }
}
