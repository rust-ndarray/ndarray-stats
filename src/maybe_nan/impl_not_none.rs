use super::NotNone;
use std::cmp;
use std::ops::{Deref, DerefMut};

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

impl<T: Eq> Eq for NotNone<T> {}

impl<T: PartialEq> PartialEq for NotNone<T> {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other)
    }
    fn ne(&self, other: &Self) -> bool {
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
