//! Shared policies for missing and non-finite numeric values.
//!
//! The initial policy surface is used by the policy-aware descriptive-summary
//! methods. It deliberately keeps missingness and infinity separate:
//!
//! | Input | Policy | Result |
//! | --- | --- | --- |
//! | finite | any policy | included |
//! | `NaN` | [`MissingDataPolicy::Propagate`] | included; normal IEEE-754 behavior applies |
//! | `NaN` | [`MissingDataPolicy::Omit`] | omitted from the calculation |
//! | `NaN` | [`MissingDataPolicy::Reject`] | returned as a typed error |
//! | `+∞` or `-∞` | [`InfinityPolicy::Propagate`] | included; normal IEEE-754 behavior applies |
//! | `+∞` or `-∞` | [`InfinityPolicy::Reject`] | returned as a typed error |
//!
//! No policy imputes, clips, or silently converts a value. In particular,
//! omission means omission of `NaN` values only; use [`InfinityPolicy::Reject`]
//! when a finite-only calculation is required. Boolean mask workflows and
//! pairwise/listwise deletion for multivariate operations remain follow-up
//! API work, while the existing `*_skipnan` methods remain available for
//! compatibility.
//!
//! A finite input is accepted, but finite inputs do not guarantee a finite
//! result: arithmetic can overflow or become indeterminate, and those results
//! follow the operation's floating-point semantics. `Propagate` means that no
//! value filtering occurs; for an order-dependent result, a NaN can therefore
//! produce the existing `UndefinedOrder` error rather than a scalar result.

/// Controls how `NaN` values are handled when they represent missing data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissingDataPolicy {
    /// Keep `NaN` values in the calculation and let the operation's documented
    /// IEEE-754 behavior apply.
    Propagate,
    /// Omit `NaN` values before calculating the result.
    ///
    /// If omission removes every value from an input or summary lane, the
    /// operation returns its ordinary empty-input error.
    Omit,
    /// Return a typed error when a `NaN` value is encountered.
    Reject,
}

/// Controls how positive and negative infinity are handled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InfinityPolicy {
    /// Include infinity and let the operation's documented IEEE-754 behavior
    /// apply. This can produce an infinite result or an indeterminate `NaN`
    /// result, such as `∞ - ∞` during variance accumulation.
    Propagate,
    /// Return a typed error when positive or negative infinity is encountered.
    Reject,
}

/// Combined missing-data and infinity policy for numeric operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NumericPolicy {
    missing_data: MissingDataPolicy,
    infinity: InfinityPolicy,
}

impl NumericPolicy {
    /// Creates a policy with independent controls for missing values and
    /// infinities.
    pub const fn new(missing_data: MissingDataPolicy, infinity: InfinityPolicy) -> Self {
        Self {
            missing_data,
            infinity,
        }
    }

    /// Returns the compatibility policy used by existing methods: NaN and
    /// infinity are propagated according to ordinary floating-point behavior.
    pub const fn propagate() -> Self {
        Self::new(MissingDataPolicy::Propagate, InfinityPolicy::Propagate)
    }

    /// Returns the standard omission policy: NaN is omitted and infinity is
    /// propagated. This is useful when NaN is the missing-value sentinel but
    /// infinity is a meaningful or diagnostically important input.
    pub const fn omit_missing() -> Self {
        Self::new(MissingDataPolicy::Omit, InfinityPolicy::Propagate)
    }

    /// Returns a finite-only policy that rejects both NaN and infinity.
    pub const fn reject_non_finite() -> Self {
        Self::new(MissingDataPolicy::Reject, InfinityPolicy::Reject)
    }

    /// Returns the missing-data behavior.
    pub const fn missing_data(self) -> MissingDataPolicy {
        self.missing_data
    }

    /// Returns the infinity behavior.
    pub const fn infinity(self) -> InfinityPolicy {
        self.infinity
    }
}

impl Default for NumericPolicy {
    fn default() -> Self {
        Self::propagate()
    }
}
