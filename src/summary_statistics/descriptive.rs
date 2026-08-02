use crate::errors::NonFiniteValue;
use crate::errors::SummaryStatisticsError;
use crate::policies::{InfinityPolicy, MissingDataPolicy, NumericPolicy};
use num_traits::{Float, FromPrimitive};
use std::cmp::Ordering;

/// A fused descriptive-statistics summary for a non-empty floating-point input.
///
/// The value stores the observation count, mean, minimum, maximum, and the
/// second central-moment accumulator used to derive population and sample
/// variance. Construct values with
/// [`crate::SummaryStatisticsExt::descriptive_statistics`] or
/// [`crate::SummaryStatisticsExt::descriptive_statistics_axis`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DescriptiveStatistics<A> {
    count: usize,
    mean: A,
    m2: A,
    min: A,
    max: A,
}

impl<A> DescriptiveStatistics<A>
where
    A: Float + FromPrimitive,
{
    /// Returns the number of observations represented by this summary.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the arithmetic mean.
    pub fn mean(&self) -> A {
        self.mean
    }

    /// Returns the minimum observation.
    pub fn min(&self) -> A {
        self.min
    }

    /// Returns the maximum observation.
    pub fn max(&self) -> A {
        self.max
    }

    /// Returns the population variance, dividing the second-moment
    /// accumulator by the number of observations.
    pub fn population_variance(&self) -> A {
        self.m2 / Self::from_usize(self.count)
    }

    /// Returns the sample variance, dividing the second-moment accumulator by
    /// one fewer than the number of observations.
    ///
    /// For a one-observation summary, this follows floating-point division
    /// semantics and is therefore `NaN`.
    pub fn sample_variance(&self) -> A {
        self.m2 / Self::from_usize(self.count - 1)
    }

    /// Returns the population standard deviation.
    pub fn population_std(&self) -> A {
        self.population_variance().sqrt()
    }

    /// Returns the sample standard deviation.
    ///
    /// For a one-observation summary, this follows floating-point division
    /// semantics and is therefore `NaN`.
    pub fn sample_std(&self) -> A {
        self.sample_variance().sqrt()
    }

    pub(super) fn from_iter<I>(values: I) -> Result<Self, SummaryStatisticsError>
    where
        I: IntoIterator<Item = A>,
    {
        Self::from_iter_with_policy(values, NumericPolicy::propagate())
    }

    pub(super) fn from_iter_with_policy<I>(
        values: I,
        policy: NumericPolicy,
    ) -> Result<Self, SummaryStatisticsError>
    where
        I: IntoIterator<Item = A>,
    {
        let mut accumulator: Option<Accumulator<A>> = None;

        for (index, value) in values.into_iter().enumerate() {
            if value.is_nan() {
                match policy.missing_data() {
                    MissingDataPolicy::Propagate => {}
                    MissingDataPolicy::Omit => continue,
                    MissingDataPolicy::Reject => {
                        return Err(SummaryStatisticsError::NonFiniteValue {
                            index,
                            value: NonFiniteValue::Nan,
                        });
                    }
                }
            } else if value.is_infinite() && policy.infinity() == InfinityPolicy::Reject {
                return Err(SummaryStatisticsError::NonFiniteValue {
                    index,
                    value: if value > A::zero() {
                        NonFiniteValue::PositiveInfinity
                    } else {
                        NonFiniteValue::NegativeInfinity
                    },
                });
            }

            if let Some(ref mut accumulator) = accumulator {
                accumulator.update(value)?;
            } else {
                accumulator = Some(Accumulator::new(value));
            }
        }

        accumulator
            .map(Accumulator::finish)
            .ok_or(SummaryStatisticsError::EmptyInput)
    }

    fn from_usize(value: usize) -> A {
        A::from_usize(value).expect("Converting an observation count to `A` must not fail.")
    }
}

struct Accumulator<A> {
    count: usize,
    mean: A,
    m2: A,
    min: A,
    max: A,
}

impl<A> Accumulator<A>
where
    A: Float + FromPrimitive,
{
    fn new(value: A) -> Self {
        Self {
            count: 1,
            mean: value,
            m2: A::zero(),
            min: value,
            max: value,
        }
    }

    fn update(&mut self, value: A) -> Result<(), SummaryStatisticsError> {
        self.count += 1;
        let count = DescriptiveStatistics::<A>::from_usize(self.count);
        let delta = value - self.mean;
        self.mean = self.mean + delta / count;
        self.m2 = self.m2 + delta * (value - self.mean);

        match value.partial_cmp(&self.min) {
            Some(Ordering::Less) => self.min = value,
            Some(_) => {}
            None => return Err(SummaryStatisticsError::UndefinedOrder),
        }
        match value.partial_cmp(&self.max) {
            Some(Ordering::Greater) => self.max = value,
            Some(_) => {}
            None => return Err(SummaryStatisticsError::UndefinedOrder),
        }

        Ok(())
    }

    fn finish(self) -> DescriptiveStatistics<A> {
        DescriptiveStatistics {
            count: self.count,
            mean: self.mean,
            m2: self.m2,
            min: self.min,
            max: self.max,
        }
    }
}
