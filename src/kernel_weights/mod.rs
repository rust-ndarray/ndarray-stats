//! Kernel weighting functions for statistical smoothing and local regression.
//!
//! This module provides common kernel functions that map a normalized
//! distance `u` (usually `|x_i - x_0| / h`) to a nonnegative weight in `[0, 1]`.
//!
//! These kernels are often used in local regression (LOESS/LOWESS),
//! kernel density estimation, and nonparametric smoothing.
//!
//! Quick reference table:
//!
//! | Kernel | Formula |
//! |---|---|
//! | Tricube | `(1 - |u|^3)^3` for `|u| < 1`, else `0` |
//! | Epanechnikov | `0.75 * (1 - u^2)` for `|u| < 1`, else `0` |
//! | Gaussian | `exp(-0.5 * u^2)` (supports all `u`) |
//! | Triangular | `1 - |u|` for `|u| < 1`, else `0` |
//! | Quartic (biweight) | `(15/16) * (1 - u^2)^2` for `|u| < 1`, else `0` |
//!
//! # Example
//! ```
//! use ndarray_stats::kernel_weights::{tricube, gaussian};
//!
//! let w1 = tricube(0.3);
//! let w2 = gaussian(0.3);
//! assert!(w1 > 0.0 && w1 <= 1.0);
//! assert!(w2 > 0.0 && w2 <= 1.0);
//! ```

/// Generic trait for kernel functions.
pub trait KernelFn {
    fn weight(&self, u: f64) -> f64;
}

// allow plain function pointers to be used as KernelFn
impl KernelFn for fn(f64) -> f64 {
    #[inline]
    fn weight(&self, u: f64) -> f64 {
        (self)(u)
    }
}

/// Tricube kernel type implementing [`KernelFn`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Tricube;
impl KernelFn for Tricube {
    #[inline]
    fn weight(&self, u: f64) -> f64 {
        tricube(u)
    }
}
pub const TRICUBE: Tricube = Tricube;

/// Gaussian kernel type implementing [`KernelFn`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Gaussian;
impl KernelFn for Gaussian {
    #[inline]
    fn weight(&self, u: f64) -> f64 {
        gaussian(u)
    }
}
pub const GAUSSIAN: Gaussian = Gaussian;

/// Epanechnikov kernel type implementing [`KernelFn`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Epanechnikov;
impl KernelFn for Epanechnikov {
    #[inline]
    fn weight(&self, u: f64) -> f64 {
        epanechnikov(u)
    }
}
pub const EPANECHNIKOV: Epanechnikov = Epanechnikov;

/// Triangular kernel type implementing [`KernelFn`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Triangular;
impl KernelFn for Triangular {
    #[inline]
    fn weight(&self, u: f64) -> f64 {
        triangular(u)
    }
}
pub const TRIANGULAR: Triangular = Triangular;

/// Quartic (biweight) kernel type implementing [`KernelFn`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Quartic;
impl KernelFn for Quartic {
    #[inline]
    fn weight(&self, u: f64) -> f64 {
        quartic(u)
    }
}
pub const QUARTIC: Quartic = Quartic;

/// Tricube kernel.
///
/// Defined as `(1 - |u|^3)^3` for `|u| < 1`, and `0` otherwise.
///
/// # Examples
/// ```
/// use ndarray_stats::kernel_weights::tricube;
/// assert_eq!(tricube(0.0), 1.0);
/// assert_eq!(tricube(1.0), 0.0);
/// ```
#[inline]
#[must_use]
pub fn tricube(u: f64) -> f64 {
    let u = u.abs();
    if u >= 1.0 {
        0.0
    } else {
        let t = 1.0 - u.powi(3);
        t.powi(3)
    }
}

/// Epanechnikov kernel.
///
/// Defined as `0.75 * (1 - u^2)` for `|u| < 1`, and `0` otherwise.
/// Optimal in a mean-square error sense for certain problems.
///
/// # Example
/// ```
/// use ndarray_stats::kernel_weights::epanechnikov;
/// assert_eq!(epanechnikov(0.0), 0.75);
/// ```
#[inline]
#[must_use]
pub fn epanechnikov(u: f64) -> f64 {
    let u = u.abs();
    if u >= 1.0 {
        0.0
    } else {
        0.75 * (1.0 - u * u)
    }
}

/// Gaussian kernel.
///
/// Defined as `exp(-0.5 * u^2)` for all real `u`.
///
/// # Example
/// ```
/// use ndarray_stats::kernel_weights::gaussian;
/// assert!((gaussian(0.0) - 1.0).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn gaussian(u: f64) -> f64 {
    (-0.5 * u * u).exp()
}

/// Triangular kernel.
///
/// Defined as `1 - |u|` for `|u| < 1`, and `0` otherwise.
/// Provides linearly decaying weights, often used in moving averages.
///
/// # Example
/// ```
/// use ndarray_stats::kernel_weights::triangular;
/// assert_eq!(triangular(0.0), 1.0);
/// assert_eq!(triangular(1.0), 0.0);
/// assert!(triangular(0.5) > 0.0);
/// ```
#[inline]
#[must_use]
pub fn triangular(u: f64) -> f64 {
    let u = u.abs();
    if u >= 1.0 {
        0.0
    } else {
        1.0 - u
    }
}

/// Quartic (biweight) kernel.
///
/// Defined as `(15/16) * (1 - u^2)^2` for `|u| < 1`, and `0` otherwise.
/// Produces a smooth, compactly supported weighting function often used
/// in kernel density estimation.
///
/// # Example
/// ```
/// use ndarray_stats::kernel_weights::quartic;
/// assert_eq!(quartic(0.0), 15.0/16.0);
/// assert_eq!(quartic(1.0), 0.0);
/// ```
#[inline]
#[must_use]
pub fn quartic(u: f64) -> f64 {
    let u = u.abs();
    if u >= 1.0 {
        0.0
    } else {
        let t = 1.0 - u * u;
        (15.0 / 16.0) * t * t
    }
}
