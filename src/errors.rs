#[derive(Fail, Debug)]
#[fail(display = "Array shapes do not match: {:?} and {:?}.", first_shape, second_shape)]
pub struct ShapeMismatch {
    pub first_shape: Vec<usize>,
    pub second_shape: Vec<usize>,
}