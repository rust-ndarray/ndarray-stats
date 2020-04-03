use num_traits::identities::{One, Zero};
use std::ops::{
  Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// Indicator for empty fields or values for binned statistic
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinContent<T>
where
  T: num_traits::Num,
{
  /// Empty bin
  Empty,
  /// Non-empty bin with some value `T`
  Value(T),
}

/// Implementation of negation operator for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// assert_eq!(-bin, BinContent::Value(-2.0));
/// ```
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
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin + bin, BinContent::<f64>::Value(4.0));
/// assert_eq!(bin + empty_bin, BinContent::Value(2.0));
/// assert_eq!(empty_bin + bin, BinContent::Value(2.0));
/// assert_eq!(empty_bin + empty_bin, BinContent::<f64>::Empty);
/// ```
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
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin += empty_bin;
/// assert_eq!(bin, BinContent::Value(2.0));
///
/// empty_bin += bin;
/// assert_eq!(empty_bin, BinContent::Value(2.0));
///
/// bin += bin;
/// assert_eq!(bin, BinContent::Value(4.0));
///
/// let mut empty_bin = BinContent::<f64>::Empty;
/// empty_bin += empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
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
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin - bin, BinContent::Value(0.0));
/// assert_eq!(bin - empty_bin, BinContent::Value(2.0));
/// assert_eq!(empty_bin - bin, BinContent::Value(-2.0));
/// assert_eq!(empty_bin - empty_bin, BinContent::<f64>::Empty);
/// ```
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
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin += empty_bin;
/// assert_eq!(bin, BinContent::Value(2.0));
///
/// empty_bin += bin;
/// assert_eq!(empty_bin, BinContent::Value(2.0));
///
/// bin += bin;
/// assert_eq!(bin, BinContent::Value(4.0));
///
/// let mut empty_bin = BinContent::<f64>::Empty;
/// empty_bin += empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
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
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin * bin, BinContent::Value(4.0));
/// assert_eq!(bin * empty_bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin * bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin * empty_bin, BinContent::<f64>::Empty);
/// ```
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
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin *= bin;
/// assert_eq!(bin, BinContent::Value(4.0));
///
/// bin *= empty_bin;
/// assert_eq!(bin, BinContent::<f64>::Empty);
///
/// let mut bin = BinContent::Value(2.0);
/// empty_bin *= bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
///
/// empty_bin *= empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
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
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin / bin, BinContent::Value(1.0));
/// assert_eq!(bin / empty_bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin / bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin / empty_bin, BinContent::<f64>::Empty);
/// ```
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
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin /= bin;
/// assert_eq!(bin, BinContent::Value(1.0));
///
/// bin /= empty_bin;
/// assert_eq!(bin, BinContent::<f64>::Empty);
///
/// let mut bin = BinContent::Value(2.0);
/// empty_bin /= bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
///
/// empty_bin /= empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
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
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let bin = BinContent::Value(3.0);
/// let den = BinContent::Value(2.0);
/// let empty_bin = BinContent::<f64>::Empty;
///
/// assert_eq!(bin % den, BinContent::Value(1.0));
/// assert_eq!(bin % empty_bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin % bin, BinContent::<f64>::Empty);
/// assert_eq!(empty_bin % empty_bin, BinContent::<f64>::Empty);
/// ```
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
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
///
/// let mut bin = BinContent::Value(3.0);
/// let mut den = BinContent::Value(2.0);
/// let mut empty_bin = BinContent::<f64>::Empty;
///
/// bin %= den;
/// assert_eq!(bin, BinContent::Value(1.0));
///
/// bin %= empty_bin;
/// assert_eq!(bin, BinContent::<f64>::Empty);
///
/// let mut bin = BinContent::Value(3.0);
/// empty_bin %= bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
///
/// empty_bin %= empty_bin;
/// assert_eq!(empty_bin, BinContent::<f64>::Empty);
/// ```
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

/// Implementation of zero-element (empty) for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
/// use num_traits::identities::Zero;
///
/// let bin = BinContent::zero();
/// assert_eq!(bin, BinContent::<f64>::Empty);
/// ```
impl<T: num_traits::Num> Zero for BinContent<T> {
  fn zero() -> Self {
    Self::Empty
  }
  fn is_zero(&self) -> bool {
    *self == Self::Empty
  }
}

/// Implementation of one-element (empty) for binned statistic indicator `BinContent`.
///
/// # Example:
/// ```
/// use ndarray_stats::histogram::BinContent;
/// use num_traits::identities::One;
///
/// let bin = BinContent::one();
/// assert_eq!(bin, BinContent::Value(1.0));
/// ```
impl<T: num_traits::Num + One> One for BinContent<T> {
  fn one() -> Self {
    Self::Value(num_traits::identities::one())
  }
  fn is_one(&self) -> bool {
    *self == Self::Value(num_traits::identities::one())
  }
}
