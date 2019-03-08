#[derive(Fail, Debug)]
#[fail(display = "Array shapes do not match: {:?} and {:?}.", first_shape, second_shape)]
/// An error used by methods and functions that take two arrays as argument and
/// expect them to have exactly the same shape
/// (e.g. `ShapeMismatch` is raised when `a.shape() == b.shape()` evaluates to `False`).
pub struct ShapeMismatch {
    pub first_shape: Vec<usize>,
    pub second_shape: Vec<usize>,
}