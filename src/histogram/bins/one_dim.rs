use std::fmt;
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
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum Bin1d<T> {
    Range(Range<T>),
    RangeFrom(RangeFrom<T>),
    RangeFull(RangeFull),
    RangeInclusive(RangeInclusive<T>),
    RangeTo(RangeTo<T>),
    RangeToInclusive(RangeToInclusive<T>),
}

impl<T> fmt::Display for Bin1d<T>
where
    T: fmt::Debug
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
    T: PartialOrd
{
    /// Return `true` if `point` belongs to the interval, `false` otherwise.
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

// Reimplemented here given that [RFC 1434](https://github.com/nox/rust-rfcs/blob/master/text/1434-contains-method-for-ranges.md)
// has not being stabilized yet and we don't want to force nightly
// for the whole library because of it
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

/// `Bins` is a collection of intervals (`Bin1d`)
/// in a 1-dimensional space.
#[derive(Debug, Clone)]
pub struct Bins1d<T> {
    bins: Vec<Bin1d<T>>,
}

impl<T> Bins1d<T>
where
    T: PartialOrd + Clone
{
    /// Given a point `P`, it returns an `Option`:
    /// - `Some(B)`, if `P` belongs to the bin `B`;
    /// - `None`, if `P` does not belong to any bin in `self`.
    ///
    /// If more than one bin in `self` contains `P`, no assumptions
    /// can be made on which bin will be returned by `find`.
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

#[cfg(test)]
mod tests {
    use super::*;
    extern crate noisy_float;

    #[test]
    fn find() {
        let bins = vec![
            Bin1d::RangeTo(..0),
            Bin1d::Range(0..5),
            Bin1d::Range(5..9),
            Bin1d::Range(10..15),
            Bin1d::RangeFrom(15..),
        ];
        let b = Bins1d { bins };
        assert_eq!(b.find(&9), None);
        assert_eq!(b.find(&15), Some(Bin1d::RangeFrom(15..)));
    }

    #[test]
    fn find_with_overlapping_bins() {
        let bins = vec![
            Bin1d::RangeToInclusive(..=0),
            Bin1d::Range(0..5),
        ];
        let b = Bins1d { bins };
        // The first one is matched and returned
        assert_eq!(b.find(&0), Some(Bin1d::RangeToInclusive(..=0)));
    }

    quickcheck! {
        fn find_with_empty_bins(point: i64) -> bool {
            let b = Bins1d { bins: vec![] };
            b.find(&point).is_none()
        }
    }
}