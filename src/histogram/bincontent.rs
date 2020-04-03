use std::ops::{
  Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// Indicator for empty fields or values for binned statistic
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinContent<T>
where
  T: num_traits::Num,
{
  Empty,
  Value(T),
}

/// Implementation of negation operator for binned statistic indicator `BinContent`.
impl<T: num_traits::Num + core::ops::Neg + Neg<Output = T>> Neg for BinContent<T> {
  type Output = Self;

  fn neg(self) -> Self {
    match self {
      BinContent::Empty => Self::Empty,
      BinContent::Value(v) => Self::Value(-v),
    }
  }
}

/// Implementation of addition for binned statistic indicator `BinContent`.
impl<T: num_traits::Num + Add<Output = T>> Add for BinContent<T> {
  type Output = Self;

  fn add(self, other: Self) -> Self {
    match (self, other) {
      (BinContent::Empty, BinContent::Empty) => Self::Empty,
      (BinContent::Value(v), BinContent::Empty) => Self::Value(v),
      (BinContent::Empty, BinContent::Value(w)) => Self::Value(w),
      (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v + w),
    }
  }
}

/// Implementation of addition assignment  for binned statistic indicator `BinContent`.
impl<T: num_traits::Num + Copy> AddAssign for BinContent<T> {
  fn add_assign(&mut self, other: Self) {
    *self = match (&self, other) {
      (BinContent::Empty, BinContent::Empty) => Self::Empty,
      (BinContent::Value(v), BinContent::Empty) => Self::Value(*v),
      (BinContent::Empty, BinContent::Value(w)) => Self::Value(w),
      (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v + *w),
    };
  }
}

/// Implementation of substraction for binned statistic indicator `BinContent`.
impl<T: num_traits::Num + Sub<Output = T>> Sub for BinContent<T> {
  type Output = Self;

  fn sub(self, other: Self) -> Self {
    match (self, other) {
      (BinContent::Empty, BinContent::Empty) => Self::Empty,
      (BinContent::Value(v), BinContent::Empty) => Self::Value(v),
      (BinContent::Empty, BinContent::Value(w)) => {
        Self::Value(num_traits::identities::zero::<T>() - w)
      }
      (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v - w),
    }
  }
}

/// Implementation of substraction assignment  for binned statistic indicator `BinContent`.
impl<T: num_traits::Num + Copy> SubAssign for BinContent<T> {
  fn sub_assign(&mut self, other: Self) {
    *self = match (&self, other) {
      (BinContent::Empty, BinContent::Empty) => Self::Empty,
      (BinContent::Value(v), BinContent::Empty) => Self::Value(*v),
      (BinContent::Empty, BinContent::Value(w)) => {
        Self::Value(num_traits::identities::zero::<T>() - w)
      }
      (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v - *w),
    };
  }
}

/// Implementation of multiplication for binned statistic indicator `BinContent`.
impl<T: num_traits::Num + Mul<Output = T>> Mul for BinContent<T> {
  type Output = Self;

  fn mul(self, other: Self) -> Self {
    match (self, other) {
      (BinContent::Empty, BinContent::Empty) => Self::Empty,
      (BinContent::Value(_), BinContent::Empty) => Self::Empty,
      (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
      (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v * w),
    }
  }
}

/// Implementation of multiplication assignment for binned statistic indicator `BinContent`.
impl<T: num_traits::Num + Copy> MulAssign for BinContent<T> {
  fn mul_assign(&mut self, other: Self) {
    *self = match (&self, other) {
      (BinContent::Empty, BinContent::Empty) => Self::Empty,
      (BinContent::Value(_), BinContent::Empty) => Self::Empty,
      (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
      (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v * *w),
    }
  }
}

/// Implementation of division for binned statistic indicator `BinContent`.
impl<T: num_traits::Num + Div<Output = T>> Div for BinContent<T> {
  type Output = Self;

  fn div(self, other: Self) -> Self {
    match (self, other) {
      (BinContent::Empty, BinContent::Empty) => Self::Empty,
      (BinContent::Value(_), BinContent::Empty) => Self::Empty,
      (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
      (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v / w),
    }
  }
}

/// Implementation of division assignment for binned statistic indicator `BinContent`.
impl<T: num_traits::Num + Copy> DivAssign for BinContent<T> {
  fn div_assign(&mut self, other: Self) {
    *self = match (&self, other) {
      (BinContent::Empty, BinContent::Empty) => Self::Empty,
      (BinContent::Value(_), BinContent::Empty) => Self::Empty,
      (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
      (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v / *w),
    }
  }
}

/// Implementation of remainder for binned statistic indicator `BinContent`.
impl<T: num_traits::Num + Div<Output = T>> Rem for BinContent<T> {
  type Output = Self;

  fn rem(self, other: Self) -> Self {
    match (self, other) {
      (BinContent::Empty, BinContent::Empty) => Self::Empty,
      (BinContent::Value(_), BinContent::Empty) => Self::Empty,
      (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
      (BinContent::Value(v), BinContent::Value(w)) => Self::Value(v % w),
    }
  }
}

/// Implementation of remainder assignment for binned statistic indicator `BinContent`.
impl<T: num_traits::Num + Copy> RemAssign for BinContent<T> {
  fn rem_assign(&mut self, other: Self) {
    *self = match (&self, other) {
      (BinContent::Empty, BinContent::Empty) => Self::Empty,
      (BinContent::Value(_), BinContent::Empty) => Self::Empty,
      (BinContent::Empty, BinContent::Value(_)) => Self::Empty,
      (BinContent::Value(v), BinContent::Value(ref w)) => Self::Value(*v % *w),
    }
  }
}
