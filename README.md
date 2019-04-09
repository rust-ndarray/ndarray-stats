# ndarray-stats

[![Build status](https://travis-ci.org/jturner314/ndarray-stats.svg?branch=master)](https://travis-ci.org/jturner314/ndarray-stats)
[![Coverage](https://codecov.io/gh/jturner314/ndarray-stats/branch/master/graph/badge.svg)](https://codecov.io/gh/jturner314/ndarray-stats)
[![Dependencies status](https://deps.rs/repo/github/jturner314/ndarray-stats/status.svg)](https://deps.rs/repo/github/jturner314/ndarray-stats)
[![Crate](https://img.shields.io/crates/v/ndarray-stats.svg)](https://crates.io/crates/ndarray-stats)
[![Documentation](https://docs.rs/ndarray-stats/badge.svg)](https://docs.rs/ndarray-stats)

This crate provides statistical methods for [`ndarray`]'s `ArrayBase` type. See
the [documentation](https://docs.rs/ndarray-stats) for more information.

[`ndarray`]: https://github.com/bluss/ndarray

Please feel free to contribute new functionality!

## Using with Cargo

```toml
[dependencies]
ndarray = "0.12.1"
ndarray-stats = "0.1"
```

## Releases

* **0.2.0**

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
  * Improvements:
    * Redesigned error handling across the whole crate (standardise on `Result` with `ErrorKind` where needed)

  *Contributors*: @jturner314, @LukeMathWalker, @phungleson, @munckymagik

* **0.1.0**

  * Initial release by @LukeMathWalker and @jturner314.

## Contributing

Please feel free to create issues and submit PRs.

## License

Copyright 2018 `ndarray-stats` developers

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE), or the [MIT
license](LICENSE-MIT), at your option. You may not use this project except in
compliance with those terms.
