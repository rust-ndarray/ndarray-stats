#[derive(Fail, Debug)]
#[fail(display = "Array shapes do not match: {:?} and {:?}.", first_shape, second_shape)]
pub struct ShapeMismatch {
    first_shape: Vec<usize>,
    second_shape: Vec<usize>,
}