use super::NotNone;
use num_traits::{FromPrimitive, ToPrimitive};
use std::cmp;
use std::fmt;
use std::ops::{Add, Deref, DerefMut, Div, Mul, Rem, Sub};

impl<T> Deref for NotNone<T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self.0 {
            Some(ref inner) => inner,
            None => unsafe { ::std::hint::unreachable_unchecked() },
        }
    }
}

impl<T> DerefMut for NotNone<T> {
    fn deref_mut(&mut self) -> &mut T {
        match self.0 {
            Some(ref mut inner) => inner,
            None => unsafe { ::std::hint::unreachable_unchecked() },
        }
    }
}

impl<T: fmt::Display> fmt::Display for NotNone<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        self.deref().fmt(f)
    }
}

impl<T: Eq> Eq for NotNone<T> {}

impl<T: PartialEq> PartialEq for NotNone<T> {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other)
    }
}

impl<T: Ord> Ord for NotNone<T> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.deref().cmp(other)
    }
}

impl<T: PartialOrd> PartialOrd for NotNone<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.deref().partial_cmp(other)
    }
    fn lt(&self, other: &Self) -> bool {
        self.deref().lt(other)
    }
    fn le(&self, other: &Self) -> bool {
        self.deref().le(other)
    }
    fn gt(&self, other: &Self) -> bool {
        self.deref().gt(other)
    }
    fn ge(&self, other: &Self) -> bool {
        self.deref().ge(other)
    }
}

impl<T: Add> Add for NotNone<T> {
    type Output = NotNone<T::Output>;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.map(|v| v.add(rhs.unwrap()))
    }
}

impl<T: Sub> Sub for NotNone<T> {
    type Output = NotNone<T::Output>;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.map(|v| v.sub(rhs.unwrap()))
    }
}

impl<T: Mul> Mul for NotNone<T> {
    type Output = NotNone<T::Output>;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.map(|v| v.mul(rhs.unwrap()))
    }
}

impl<T: Div> Div for NotNone<T> {
    type Output = NotNone<T::Output>;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.map(|v| v.div(rhs.unwrap()))
    }
}

impl<T: Rem> Rem for NotNone<T> {
    type Output = NotNone<T::Output>;
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        self.map(|v| v.rem(rhs.unwrap()))
    }
}

impl<T: ToPrimitive> ToPrimitive for NotNone<T> {
    #[inline]
    fn to_isize(&self) -> Option<isize> {
        self.deref().to_isize()
    }
    #[inline]
    fn to_i8(&self) -> Option<i8> {
        self.deref().to_i8()
    }
    #[inline]
    fn to_i16(&self) -> Option<i16> {
        self.deref().to_i16()
    }
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        self.deref().to_i32()
    }
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.deref().to_i64()
    }
    #[inline]
    fn to_i128(&self) -> Option<i128> {
        self.deref().to_i128()
    }
    #[inline]
    fn to_usize(&self) -> Option<usize> {
        self.deref().to_usize()
    }
    #[inline]
    fn to_u8(&self) -> Option<u8> {
        self.deref().to_u8()
    }
    #[inline]
    fn to_u16(&self) -> Option<u16> {
        self.deref().to_u16()
    }
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        self.deref().to_u32()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.deref().to_u64()
    }
    #[inline]
    fn to_u128(&self) -> Option<u128> {
        self.deref().to_u128()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.deref().to_f32()
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.deref().to_f64()
    }
}

impl<T: FromPrimitive> FromPrimitive for NotNone<T> {
    #[inline]
    fn from_isize(n: isize) -> Option<Self> {
        Self::try_new(T::from_isize(n))
    }
    #[inline]
    fn from_i8(n: i8) -> Option<Self> {
        Self::try_new(T::from_i8(n))
    }
    #[inline]
    fn from_i16(n: i16) -> Option<Self> {
        Self::try_new(T::from_i16(n))
    }
    #[inline]
    fn from_i32(n: i32) -> Option<Self> {
        Self::try_new(T::from_i32(n))
    }
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        Self::try_new(T::from_i64(n))
    }
    #[inline]
    fn from_i128(n: i128) -> Option<Self> {
        Self::try_new(T::from_i128(n))
    }
    #[inline]
    fn from_usize(n: usize) -> Option<Self> {
        Self::try_new(T::from_usize(n))
    }
    #[inline]
    fn from_u8(n: u8) -> Option<Self> {
        Self::try_new(T::from_u8(n))
    }
    #[inline]
    fn from_u16(n: u16) -> Option<Self> {
        Self::try_new(T::from_u16(n))
    }
    #[inline]
    fn from_u32(n: u32) -> Option<Self> {
        Self::try_new(T::from_u32(n))
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        Self::try_new(T::from_u64(n))
    }
    #[inline]
    fn from_u128(n: u128) -> Option<Self> {
        Self::try_new(T::from_u128(n))
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        Self::try_new(T::from_f32(n))
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        Self::try_new(T::from_f64(n))
    }
}
