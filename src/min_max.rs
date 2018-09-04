use ndarray::prelude::*;
use ndarray::Data;
use std::cmp::Ordering;
use {MaybeNan, MaybeNanExt};

/// Minimum and maximum methods.
pub trait MinMaxExt<A, D: Dimension> {
    /// Finds the elementwise minimum of the array.
    ///
    /// **Panics** if the array is empty.
    fn min(&self) -> &A
    where
        A: Ord;

    /// Finds the elementwise minimum of the array.
    ///
    /// Returns `None` if any of the pairwise orderings tested by the function
    /// are undefined. (For example, this occurs if there are any
    /// floating-point NaN values in the array.)
    ///
    /// Additionally, returns `None` if the array is empty.
    fn min_partialord(&self) -> Option<&A>
    where
        A: PartialOrd;

    /// Finds the elementwise minimum of the array, skipping NaN values.
    ///
    /// **Warning** This method will return a NaN value if none of the values
    /// in the array are non-NaN values. Note that the NaN value might not be
    /// in the array.
    fn min_skipnan(&self) -> &A
    where
        A: MaybeNan,
        A::NotNan: Ord;

    /// Finds the elementwise maximum of the array.
    ///
    /// **Panics** if the array is empty.
    fn max(&self) -> &A
    where
        A: Ord;

    /// Finds the elementwise maximum of the array.
    ///
    /// Returns `None` if any of the pairwise orderings tested by the function
    /// are undefined. (For example, this occurs if there are any
    /// floating-point NaN values in the array.)
    ///
    /// Additionally, returns `None` if the array is empty.
    fn max_partialord(&self) -> Option<&A>
    where
        A: PartialOrd;

    /// Finds the elementwise maximum of the array, skipping NaN values.
    ///
    /// **Warning** This method will return a NaN value if none of the values
    /// in the array are non-NaN values. Note that the NaN value might not be
    /// in the array.
    fn max_skipnan(&self) -> &A
    where
        A: MaybeNan,
        A::NotNan: Ord;
}

impl<A, S, D> MinMaxExt<A, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn min(&self) -> &A
    where
        A: Ord,
    {
        let first = self
            .iter()
            .next()
            .expect("Attempted to find min of empty array.");
        self.fold(first, |acc, elem| if elem < acc { elem } else { acc })
    }

    fn min_partialord(&self) -> Option<&A>
    where
        A: PartialOrd,
    {
        let first = self.iter().next()?;
        self.fold(Some(first), |acc, elem| match elem.partial_cmp(acc?)? {
            Ordering::Less => Some(elem),
            _ => acc,
        })
    }

    fn min_skipnan(&self) -> &A
    where
        A: MaybeNan,
        A::NotNan: Ord,
    {
        let first = self.iter().next().and_then(|v| v.try_as_not_nan());
        A::from_opt_not_nan(self.fold_skipnan(first, |acc, elem| {
            Some(match acc {
                Some(acc) => acc.min(elem),
                None => elem,
            })
        }))
    }

    fn max(&self) -> &A
    where
        A: Ord,
    {
        let first = self
            .iter()
            .next()
            .expect("Attempted to find max of empty array.");
        self.fold(first, |acc, elem| if elem > acc { elem } else { acc })
    }

    fn max_partialord(&self) -> Option<&A>
    where
        A: PartialOrd,
    {
        let first = self.iter().next()?;
        self.fold(Some(first), |acc, elem| match elem.partial_cmp(acc?)? {
            Ordering::Greater => Some(elem),
            _ => acc,
        })
    }

    fn max_skipnan(&self) -> &A
    where
        A: MaybeNan,
        A::NotNan: Ord,
    {
        let first = self.iter().next().and_then(|v| v.try_as_not_nan());
        A::from_opt_not_nan(self.fold_skipnan(first, |acc, elem| {
            Some(match acc {
                Some(acc) => acc.max(elem),
                None => elem,
            })
        }))
    }
}
