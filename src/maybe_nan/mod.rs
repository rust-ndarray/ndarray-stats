use ndarray::prelude::*;
use ndarray::{s, Data, DataMut, RemoveAxis};
use noisy_float::types::{N32, N64};
use std::mem;

/// A number type that can have not-a-number values.
pub trait MaybeNan: Sized {
    /// A type that is guaranteed not to be a NaN value.
    type NotNan;

    /// Returns `true` if the value is a NaN value.
    fn is_nan(&self) -> bool;

    /// Tries to convert the value to `NotNan`.
    ///
    /// Returns `None` if the value is a NaN value.
    fn try_as_not_nan(&self) -> Option<&Self::NotNan>;

    /// Converts the value.
    ///
    /// If the value is `None`, a NaN value is returned.
    fn from_not_nan(_: Self::NotNan) -> Self;

    /// Converts the value.
    ///
    /// If the value is `None`, a NaN value is returned.
    fn from_not_nan_opt(_: Option<Self::NotNan>) -> Self;

    /// Converts the value.
    ///
    /// If the value is `None`, a NaN value is returned.
    fn from_not_nan_ref_opt(_: Option<&Self::NotNan>) -> &Self;

    /// Returns a view with the NaN values removed.
    ///
    /// This modifies the input view by moving elements as necessary. The final
    /// order of the elements is unspecified. However, this method is
    /// idempotent, and given the same input data, the result is always ordered
    /// the same way.
    fn remove_nan_mut(_: ArrayViewMut1<'_, Self>) -> ArrayViewMut1<'_, Self::NotNan>;
}

/// Returns a view with the NaN values removed.
///
/// This modifies the input view by moving elements as necessary.
fn remove_nan_mut<A: MaybeNan>(mut view: ArrayViewMut1<'_, A>) -> ArrayViewMut1<'_, A> {
    if view.is_empty() {
        return view.slice_move(s![..0]);
    }
    let mut i = 0;
    let mut j = view.len() - 1;
    loop {
        // At this point, `i == 0 || !view[i-1].is_nan()`
        // and `j == view.len() - 1 || view[j+1].is_nan()`.
        while i <= j && !view[i].is_nan() {
            i += 1;
        }
        // At this point, `view[i].is_nan() || i == j + 1`.
        while j > i && view[j].is_nan() {
            j -= 1;
        }
        // At this point, `!view[j].is_nan() || j == i`.
        if i >= j {
            return view.slice_move(s![..i]);
        } else {
            view.swap(i, j);
            i += 1;
            j -= 1;
        }
    }
}

/// Casts a view from one element type to another.
///
/// # Panics
///
/// Panics if `T` and `U` differ in size or alignment.
///
/// # Safety
///
/// The caller must ensure that qll elements in `view` are valid values for type `U`.
unsafe fn cast_view_mut<T, U>(mut view: ArrayViewMut1<'_, T>) -> ArrayViewMut1<'_, U> {
    assert_eq!(mem::size_of::<T>(), mem::size_of::<U>());
    assert_eq!(mem::align_of::<T>(), mem::align_of::<U>());
    let ptr: *mut U = view.as_mut_ptr().cast();
    let len: usize = view.len_of(Axis(0));
    let stride: isize = view.stride_of(Axis(0));
    if len <= 1 {
        // We can use a stride of `0` because the stride is irrelevant for the `len == 1` case.
        let stride = 0;
        ArrayViewMut1::from_shape_ptr([len].strides([stride]), ptr)
    } else if stride >= 0 {
        let stride = stride as usize;
        ArrayViewMut1::from_shape_ptr([len].strides([stride]), ptr)
    } else {
        // At this point, stride < 0. We have to construct the view by using the inverse of the
        // stride and then inverting the axis, since `ArrayViewMut::from_shape_ptr` requires the
        // stride to be nonnegative.
        let neg_stride = stride.checked_neg().unwrap() as usize;
        // This is safe because `ndarray` guarantees that it's safe to offset the
        // pointer anywhere in the array.
        let neg_ptr = ptr.offset((len - 1) as isize * stride);
        let mut v = ArrayViewMut1::from_shape_ptr([len].strides([neg_stride]), neg_ptr);
        v.invert_axis(Axis(0));
        v
    }
}

macro_rules! impl_maybenan_for_fxx {
    ($fxx:ident, $Nxx:ident) => {
        impl MaybeNan for $fxx {
            type NotNan = $Nxx;

            fn is_nan(&self) -> bool {
                $fxx::is_nan(*self)
            }

            fn try_as_not_nan(&self) -> Option<&$Nxx> {
                $Nxx::try_borrowed(self)
            }

            fn from_not_nan(value: $Nxx) -> $fxx {
                value.raw()
            }

            fn from_not_nan_opt(value: Option<$Nxx>) -> $fxx {
                match value {
                    None => ::std::$fxx::NAN,
                    Some(num) => num.raw(),
                }
            }

            fn from_not_nan_ref_opt(value: Option<&$Nxx>) -> &$fxx {
                match value {
                    None => &::std::$fxx::NAN,
                    Some(num) => num.as_ref(),
                }
            }

            fn remove_nan_mut(view: ArrayViewMut1<'_, $fxx>) -> ArrayViewMut1<'_, $Nxx> {
                let not_nan = remove_nan_mut(view);
                // This is safe because `remove_nan_mut` has removed the NaN values, and `$Nxx` is
                // a thin wrapper around `$fxx`.
                unsafe { cast_view_mut(not_nan) }
            }
        }
    };
}
impl_maybenan_for_fxx!(f32, N32);
impl_maybenan_for_fxx!(f64, N64);

macro_rules! impl_maybenan_for_opt_never_nan {
    ($ty:ty) => {
        impl MaybeNan for Option<$ty> {
            type NotNan = NotNone<$ty>;

            fn is_nan(&self) -> bool {
                self.is_none()
            }

            fn try_as_not_nan(&self) -> Option<&NotNone<$ty>> {
                if self.is_none() {
                    None
                } else {
                    // This is safe because we have checked for the `None`
                    // case, and `NotNone<$ty>` is a thin wrapper around `Option<$ty>`.
                    Some(unsafe { &*(self as *const Option<$ty> as *const NotNone<$ty>) })
                }
            }

            fn from_not_nan(value: NotNone<$ty>) -> Option<$ty> {
                value.into_inner()
            }

            fn from_not_nan_opt(value: Option<NotNone<$ty>>) -> Option<$ty> {
                value.and_then(|v| v.into_inner())
            }

            fn from_not_nan_ref_opt(value: Option<&NotNone<$ty>>) -> &Option<$ty> {
                match value {
                    None => &None,
                    // This is safe because `NotNone<$ty>` is a thin wrapper around
                    // `Option<$ty>`.
                    Some(num) => unsafe { &*(num as *const NotNone<$ty> as *const Option<$ty>) },
                }
            }

            fn remove_nan_mut(view: ArrayViewMut1<'_, Self>) -> ArrayViewMut1<'_, Self::NotNan> {
                let not_nan = remove_nan_mut(view);
                // This is safe because `remove_nan_mut` has removed the `None`
                // values, and `NotNone<$ty>` is a thin wrapper around `Option<$ty>`.
                unsafe {
                    ArrayViewMut1::from_shape_ptr(
                        not_nan.dim(),
                        not_nan.as_ptr() as *mut NotNone<$ty>,
                    )
                }
            }
        }
    };
}
impl_maybenan_for_opt_never_nan!(u8);
impl_maybenan_for_opt_never_nan!(u16);
impl_maybenan_for_opt_never_nan!(u32);
impl_maybenan_for_opt_never_nan!(u64);
impl_maybenan_for_opt_never_nan!(u128);
impl_maybenan_for_opt_never_nan!(i8);
impl_maybenan_for_opt_never_nan!(i16);
impl_maybenan_for_opt_never_nan!(i32);
impl_maybenan_for_opt_never_nan!(i64);
impl_maybenan_for_opt_never_nan!(i128);
impl_maybenan_for_opt_never_nan!(N32);
impl_maybenan_for_opt_never_nan!(N64);

/// A thin wrapper around `Option` that guarantees that the value is not
/// `None`.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct NotNone<T>(Option<T>);

impl<T> NotNone<T> {
    /// Creates a new `NotNone` containing the given value.
    pub fn new(value: T) -> NotNone<T> {
        NotNone(Some(value))
    }

    /// Creates a new `NotNone` containing the given value.
    ///
    /// Returns `None` if `value` is `None`.
    pub fn try_new(value: Option<T>) -> Option<NotNone<T>> {
        if value.is_some() {
            Some(NotNone(value))
        } else {
            None
        }
    }

    /// Returns the underling option.
    pub fn into_inner(self) -> Option<T> {
        self.0
    }

    /// Moves the value out of the inner option.
    ///
    /// This method is guaranteed not to panic.
    pub fn unwrap(self) -> T {
        match self.0 {
            Some(inner) => inner,
            None => unsafe { ::std::hint::unreachable_unchecked() },
        }
    }

    /// Maps an `NotNone<T>` to `NotNone<U>` by applying a function to the
    /// contained value.
    pub fn map<U, F>(self, f: F) -> NotNone<U>
    where
        F: FnOnce(T) -> U,
    {
        NotNone::new(f(self.unwrap()))
    }
}

/// Extension trait for `ArrayBase` providing NaN-related functionality.
pub trait MaybeNanExt<A, S, D>
where
    A: MaybeNan,
    S: Data<Elem = A>,
    D: Dimension,
{
    /// Traverse the non-NaN array elements and apply a fold, returning the
    /// resulting value.
    ///
    /// Elements are visited in arbitrary order.
    fn fold_skipnan<'a, F, B>(&'a self, init: B, f: F) -> B
    where
        A: 'a,
        F: FnMut(B, &'a A::NotNan) -> B;

    /// Traverse the non-NaN elements and their indices and apply a fold,
    /// returning the resulting value.
    ///
    /// Elements are visited in arbitrary order.
    fn indexed_fold_skipnan<'a, F, B>(&'a self, init: B, f: F) -> B
    where
        A: 'a,
        F: FnMut(B, (D::Pattern, &'a A::NotNan)) -> B;

    /// Visit each non-NaN element in the array by calling `f` on each element.
    ///
    /// Elements are visited in arbitrary order.
    fn visit_skipnan<'a, F>(&'a self, f: F)
    where
        A: 'a,
        F: FnMut(&'a A::NotNan);

    /// Fold non-NaN values along an axis.
    ///
    /// Combine the non-NaN elements of each subview with the previous using
    /// the fold function and initial value init.
    fn fold_axis_skipnan<B, F>(&self, axis: Axis, init: B, fold: F) -> Array<B, D::Smaller>
    where
        D: RemoveAxis,
        F: FnMut(&B, &A::NotNan) -> B,
        B: Clone;

    /// Reduce the values along an axis into just one value, producing a new
    /// array with one less dimension.
    ///
    /// The NaN values are removed from the 1-dimensional lanes, then they are
    /// passed as mutable views to the reducer, allowing for side-effects.
    ///
    /// **Warnings**:
    ///
    /// * The lanes are visited in arbitrary order.
    ///
    /// * The order of the elements within the lanes is unspecified. However,
    ///   if `mapping` is idempotent, this method is idempotent. Additionally,
    ///   given the same input data, the lane is always ordered the same way.
    ///
    /// **Panics** if `axis` is out of bounds.
    fn map_axis_skipnan_mut<'a, B, F>(&'a mut self, axis: Axis, mapping: F) -> Array<B, D::Smaller>
    where
        A: 'a,
        S: DataMut,
        D: RemoveAxis,
        F: FnMut(ArrayViewMut1<'a, A::NotNan>) -> B;

    private_decl! {}
}

impl<A, S, D> MaybeNanExt<A, S, D> for ArrayBase<S, D>
where
    A: MaybeNan,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn fold_skipnan<'a, F, B>(&'a self, init: B, mut f: F) -> B
    where
        A: 'a,
        F: FnMut(B, &'a A::NotNan) -> B,
    {
        self.fold(init, |acc, elem| {
            if let Some(not_nan) = elem.try_as_not_nan() {
                f(acc, not_nan)
            } else {
                acc
            }
        })
    }

    fn indexed_fold_skipnan<'a, F, B>(&'a self, init: B, mut f: F) -> B
    where
        A: 'a,
        F: FnMut(B, (D::Pattern, &'a A::NotNan)) -> B,
    {
        self.indexed_iter().fold(init, |acc, (idx, elem)| {
            if let Some(not_nan) = elem.try_as_not_nan() {
                f(acc, (idx, not_nan))
            } else {
                acc
            }
        })
    }

    fn visit_skipnan<'a, F>(&'a self, mut f: F)
    where
        A: 'a,
        F: FnMut(&'a A::NotNan),
    {
        self.for_each(|elem| {
            if let Some(not_nan) = elem.try_as_not_nan() {
                f(not_nan)
            }
        })
    }

    fn fold_axis_skipnan<B, F>(&self, axis: Axis, init: B, mut fold: F) -> Array<B, D::Smaller>
    where
        D: RemoveAxis,
        F: FnMut(&B, &A::NotNan) -> B,
        B: Clone,
    {
        self.fold_axis(axis, init, |acc, elem| {
            if let Some(not_nan) = elem.try_as_not_nan() {
                fold(acc, not_nan)
            } else {
                acc.clone()
            }
        })
    }

    fn map_axis_skipnan_mut<'a, B, F>(
        &'a mut self,
        axis: Axis,
        mut mapping: F,
    ) -> Array<B, D::Smaller>
    where
        A: 'a,
        S: DataMut,
        D: RemoveAxis,
        F: FnMut(ArrayViewMut1<'a, A::NotNan>) -> B,
    {
        self.map_axis_mut(axis, |lane| mapping(A::remove_nan_mut(lane)))
    }

    private_impl! {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn remove_nan_mut_idempotent(is_nan: Vec<bool>) -> bool {
        let mut values: Vec<_> = is_nan
            .into_iter()
            .map(|is_nan| if is_nan { None } else { Some(1) })
            .collect();
        let view = ArrayViewMut1::from_shape(values.len(), &mut values).unwrap();
        let removed = remove_nan_mut(view);
        removed == remove_nan_mut(removed.to_owned().view_mut())
    }

    #[quickcheck]
    fn remove_nan_mut_only_nan_remaining(is_nan: Vec<bool>) -> bool {
        let mut values: Vec<_> = is_nan
            .into_iter()
            .map(|is_nan| if is_nan { None } else { Some(1) })
            .collect();
        let view = ArrayViewMut1::from_shape(values.len(), &mut values).unwrap();
        remove_nan_mut(view).iter().all(|elem| !elem.is_nan())
    }

    #[quickcheck]
    fn remove_nan_mut_keep_all_non_nan(is_nan: Vec<bool>) -> bool {
        let non_nan_count = is_nan.iter().filter(|&&is_nan| !is_nan).count();
        let mut values: Vec<_> = is_nan
            .into_iter()
            .map(|is_nan| if is_nan { None } else { Some(1) })
            .collect();
        let view = ArrayViewMut1::from_shape(values.len(), &mut values).unwrap();
        remove_nan_mut(view).len() == non_nan_count
    }
}

mod impl_not_none;
