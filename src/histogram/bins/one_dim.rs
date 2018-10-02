use std::fmt;
use std::hash::Hash;
use std::ops::{Bound, Range, RangeBounds, RangeFrom, RangeFull,
               RangeInclusive, RangeTo, RangeToInclusive};

/// One-dimensional intervals.
///
/// # Example
///
/// ```
/// extern crate ndarray_stats;
/// extern crate noisy_float;
/// use ndarray_stats::Bin1d;
/// use noisy_float::types::n64;
///
/// let unit_interval = Bin1d::RangeInclusive(n64(0.)..=n64(1.));
/// assert!(unit_interval.contains(&n64(1.)));
/// assert!(unit_interval.contains(&n64(0.)));
/// assert!(unit_interval.contains(&n64(0.5)));
/// ```
#[derive(Hash, PartialEq, Eq, Clone)]
pub enum Bin1d<T: Hash + Eq + Clone> {
    Range(Range<T>),
    RangeFrom(RangeFrom<T>),
    RangeFull(RangeFull),
    RangeInclusive(RangeInclusive<T>),
    RangeTo(RangeTo<T>),
    RangeToInclusive(RangeToInclusive<T>),
}

impl<T> fmt::Display for Bin1d<T>
where
    T: Hash + Eq + Clone + fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Bin1d::Range(x) => write!(f, "{:?}", x),
            Bin1d::RangeFrom(x) => write!(f, "{:?}", x),
            Bin1d::RangeFull(x) => write!(f, "{:?}", x),
            Bin1d::RangeInclusive(x) => write!(f, "{:?}", x),
            Bin1d::RangeTo(x) => write!(f, "{:?}", x),
            Bin1d::RangeToInclusive(x) => write!(f, "{:?}", x),
        }
    }
}

impl<T> Bin1d<T>
where
    T: Hash + Eq + Clone + PartialOrd
{
    pub fn contains(&self, point: &T) -> bool
    {
        match self {
            Bin1d::Range(x) => contains::<Range<T>, T>(x, point),
            Bin1d::RangeFrom(x) => contains::<RangeFrom<T>, T>(x, point),
            Bin1d::RangeFull(_) => true,
            Bin1d::RangeInclusive(x) => contains::<RangeInclusive<T>, T>(x, point),
            Bin1d::RangeTo(x) => contains::<RangeTo<T>, T>(x, point),
            Bin1d::RangeToInclusive(x) => contains::<RangeToInclusive<T>, T>(x, point),
        }
    }
}

fn contains<R, T>(range: &R, item: &T) -> bool
where
    R: RangeBounds<T>,
    T: PartialOrd,
{
    (match range.start_bound() {
        Bound::Included(ref start) => *start <= item,
        Bound::Excluded(ref start) => *start < item,
        Bound::Unbounded => true,
    })
    &&
    (match range.end_bound() {
        Bound::Included(ref end) => item <= *end,
        Bound::Excluded(ref end) => item < *end,
        Bound::Unbounded => true,
    })
}

/// `Bins` is a collection of non-overlapping
/// intervals (`Bin1d`) in a 1-dimensional space.
pub struct Bins1d<T: Hash + Eq + Clone> {
    bins: Vec<Bin1d<T>>,
}

impl<T> Bins1d<T>
where
    T: Hash + Eq + Clone + PartialOrd
{
    /// Given a point `P`, it returns an `Option`:
    /// - `Some(B)`, if `P` belongs to the `Bin1d` `B`;
    /// - `None`, if `P` does not belong to any `Bin1d` in `Bins`.
    pub fn find(&self, point: &T) -> Option<Bin1d<T>>
    {
        for bin in self.bins.iter() {
            if bin.contains(point) {
                return Some((*bin).clone())
            }
        }
        None
    }
}