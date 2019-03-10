use std::error::Error;
use std::fmt;

#[derive(Debug)]
/// An error used by methods and functions that take two arrays as argument and
/// expect them to have exactly the same shape
/// (e.g. `ShapeMismatch` is raised when `a.shape() == b.shape()` evaluates to `False`).
pub struct ShapeMismatch {
    pub first_shape: Vec<usize>,
    pub second_shape: Vec<usize>,
}

impl fmt::Display for ShapeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Array shapes do not match: {:?} and {:?}.",
            self.first_shape, self.second_shape
        )
    }
}

impl Error for ShapeMismatch {}
