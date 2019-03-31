use crate::errors::{EmptyInput, MinMaxError};
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
}

/// Error computing the set of histogram bins according to a specific strategy.
#[derive(Debug, Clone)]
pub enum StrategyError {
    /// The input array was empty.
    EmptyInput,
    /// Other error.
    Other,
}

impl StrategyError {
    /// Returns whether `self` is the `EmptyInput` variant.
    pub fn is_empty_input(&self) -> bool {
        match self {
            StrategyError::EmptyInput => true,
            _ => false,
        }
    }

    /// Returns whether `self` is the `Other` variant.
    pub fn is_other(&self) -> bool {
        match self {
            StrategyError::Other => true,
            _ => false,
        }
    }
}

impl fmt::Display for StrategyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "The strategy failed to determine a non-zero bin width.")
    }
}

impl error::Error for StrategyError {
    fn description(&self) -> &str {
        "The strategy failed to determine a non-zero bin width."
    }
}

impl From<EmptyInput> for StrategyError {
    fn from(_: EmptyInput) -> Self {
        StrategyError::EmptyInput
    }
}

impl From<MinMaxError> for StrategyError {
    fn from(err: MinMaxError) -> StrategyError {
        match err {
            MinMaxError::EmptyInput => StrategyError::EmptyInput,
            MinMaxError::UndefinedOrder => StrategyError::Other,
        }
    }
}
