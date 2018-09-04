# ndarray-stats

This crate provides statistical methods for [`ndarray`]'s `ArrayBase` type.

[`ndarray`]: https://github.com/bluss/ndarray

Only some statistical functions are implemented. Please feel free to contribute
new functionality!

## Using with Cargo

```toml
[dependencies]
ndarray = "0.12"
ndarray-stats = "0.1"
```

## Releases

* **0.1.0** (not yet released)

  * Initial release.
  * Includes quantile functionality provided by @LukeMathWalker. (See
    [`ndarray` issue #461](https://github.com/bluss/ndarray/pull/461).)

## Contributing

Please feel free to create issues and submit PRs.

## License

Copyright 2018 `ndarray-stats` developers

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE), or the [MIT
license](LICENSE-MIT), at your option. You may not use this project except in
compliance with those terms.
