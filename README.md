# ndarray-stats

[![Build status](https://travis-ci.org/rust-ndarray/ndarray-stats.svg?branch=master)](https://travis-ci.org/rust-ndarray/ndarray-stats)
[![Coverage](https://codecov.io/gh/rust-ndarray/ndarray-stats/branch/master/graph/badge.svg)](https://codecov.io/gh/rust-ndarray/ndarray-stats)
[![Dependencies status](https://deps.rs/repo/github/rust-ndarray/ndarray-stats/status.svg)](https://deps.rs/repo/github/rust-ndarray/ndarray-stats)
[![Crate](https://img.shields.io/crates/v/ndarray-stats.svg)](https://crates.io/crates/ndarray-stats)
[![Documentation](https://docs.rs/ndarray-stats/badge.svg)](https://docs.rs/ndarray-stats)

This crate provides statistical methods for [`ndarray`]'s `ArrayBase` type.

Currently available routines include:
- order statistics (minimum, maximum, median, quantiles, etc.);
- summary statistics (mean, skewness, kurtosis, central moments, etc.)
- partitioning;
- correlation analysis (covariance, pearson correlation);
- measures from information theory (entropy, KL divergence, etc.);
- deviation functions (distances, counts, errors, etc.);
- histogram computation.

See the [documentation](https://docs.rs/ndarray-stats) for more information.

Please feel free to contribute new functionality! A roadmap can be found [here](https://github.com/rust-ndarray/ndarray-stats/issues/1).

[`ndarray`]: https://github.com/rust-ndarray/ndarray

## Using with Cargo

```toml
[dependencies]
ndarray = "0.14"
ndarray-stats = "0.4"
```

## Releases

* **0.5.0**
  * Breaking changes
    * Minimum supported Rust version: `1.49.0`
    * Updated to `ndarray:v0.15.0`

  *Contributors*: [@Armavica](https://github.com/armavica), [@cassiersg](https://github.com/cassiersg)

* **0.4.0**
  * Breaking changes
    * Minimum supported Rust version: `1.42.0`
  * New functionality:
    * Summary statistics:
      * Weighted variance
      * Weighted standard deviation
  * Improvements / breaking changes:
    * Documentation improvements for Histograms
    * Updated to `ndarray:v0.14.0`
 
  *Contributors*: [@munckymagik](https://github.com/munckymagik), [@nilgoyette](https://github.com/nilgoyette), [@LukeMathWalker](https://github.com/LukeMathWalker), [@lebensterben](https://github.com/lebensterben), [@xd009642](https://github.com/xd009642)

* **0.3.0**

  * Breaking changes
    * Minimum supported Rust version: `1.37`
  * New functionality:
    * Deviation functions:
      * Counts equal/unequal
      * `l1`, `l2`, `linf` distances
      * (Root) mean squared error
      * Peak signal-to-noise ratio
    * Summary statistics:
      * Weighted sum
      * Weighted mean
  * Improvements / breaking changes:
    * Updated to `ndarray:v0.13.0`
  
  *Contributors*: [@munckymagik](https://github.com/munckymagik), [@nilgoyette](https://github.com/nilgoyette), [@jturner314](https://github.com/jturner314), [@LukeMathWalker](https://github.com/LukeMathWalker)

* **0.2.0**

  * Breaking changes
    * All `ndarray-stats`' extension traits are now impossible to implement by
      users of the library (see [#34])
    * Redesigned error handling across the whole crate, standardising on `Result`
  * New functionality:
    * Summary statistics:
      * Harmonic mean
      * Geometric mean
      * Central moments
      * Kurtosis
      * Skewness
    * Information theory:
      * Entropy
      * Cross-entropy
      * Kullback-Leibler divergence
    * Quantiles and order statistics:
      * `argmin` / `argmin_skipnan`
      * `argmax` / `argmax_skipnan`
      * Optimized bulk quantile computation (`quantiles_mut`, `quantiles_axis_mut`)
  * Fixes:
    * Reduced occurrences of overflow for `interpolate::midpoint`

  *Contributors*: [@jturner314](https://github.com/jturner314), [@LukeMathWalker](https://github.com/LukeMathWalker), [@phungleson](https://github.com/phungleson), [@munckymagik](https://github.com/munckymagik)

  [#34]: https://github.com/rust-ndarray/ndarray-stats/issues/34

* **0.1.0**

  * Initial release by @LukeMathWalker and @jturner314.

## Contributing

Please feel free to create issues and submit PRs.

## License

Copyright 2018 `ndarray-stats` developers

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE), or the [MIT
license](LICENSE-MIT), at your option. You may not use this project except in
compliance with those terms.
