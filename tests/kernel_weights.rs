use ndarray_stats::kernel_weights::*;

fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = 0.0;
    for i in 0..n {
        let x0 = a + i as f64 * h;
        let x1 = a + (i + 1) as f64 * h;
        sum += 0.5 * (f(x0) + f(x1)) * h;
    }
    sum
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn tricube_basic_properties() {
    assert!(approx_eq(tricube(0.5), tricube(-0.5), 1e-15));
    assert_eq!(tricube(1.0), 0.0);
    assert_eq!(tricube(-1.0), 0.0);
    assert_eq!(tricube(0.0), 1.0);
    assert!(tricube(0.25) > tricube(0.5));
    assert!(tricube(0.5) > tricube(0.75));
}

#[test]
fn epanechnikov_behavior() {
    assert!(approx_eq(epanechnikov(0.3), epanechnikov(-0.3), 1e-15));
    assert_eq!(epanechnikov(1.0), 0.0);
    assert_eq!(epanechnikov(-1.0), 0.0);
    assert!(epanechnikov(0.0) > epanechnikov(0.8));
    assert!(epanechnikov(0.5) > 0.0);
    assert!(epanechnikov(0.5) < epanechnikov(0.0));
}

#[test]
fn quartic_behavior() {
    assert!(approx_eq(quartic(0.0), 15.0 / 16.0, 1e-12));
    assert_eq!(quartic(1.0), 0.0);
    assert_eq!(quartic(-1.0), 0.0);
    assert!(approx_eq(quartic(0.3), quartic(-0.3), 1e-15));
    assert!(quartic(0.25) > quartic(0.75));
    assert_eq!(quartic(1.1), 0.0);
}

#[test]
fn triangular_behavior() {
    assert_eq!(triangular(0.0), 1.0);
    assert_eq!(triangular(1.0), 0.0);
    assert_eq!(triangular(-1.0), 0.0);
    assert!(approx_eq(triangular(0.3), triangular(-0.3), 1e-15));
    assert!(triangular(0.25) > triangular(0.75));
    assert_eq!(triangular(1.2), 0.0);
}

#[test]
fn gaussian_behavior() {
    assert!(approx_eq(gaussian(0.5), gaussian(-0.5), 1e-15));
    assert!(gaussian(0.0) > gaussian(1.0));
    assert!(gaussian(2.0) < 0.2);
    for u in [-3.0, -1.0, 0.0, 1.0, 3.0] {
        assert!(gaussian(u) >= 0.0);
    }
}

#[test]
fn kernel_trait_usage() {
    let t = Tricube;
    let g = Gaussian;
    assert_eq!(t.weight(0.0), 1.0);
    assert!(g.weight(1.0) < 1.0);
    assert!(approx_eq(t.weight(0.5), tricube(0.5), 1e-15));
    struct Linear;
    impl KernelFn for Linear {
        fn weight(&self, u: f64) -> f64 {
            (1.0 - u.abs()).max(0.0)
        }
    }
    let lin = Linear;
    assert_eq!(lin.weight(0.0), 1.0);
    assert_eq!(lin.weight(1.5), 0.0);
}

#[test]
fn kernel_types_and_consts() {
    assert_eq!(TRICUBE, Tricube::default());
    assert_eq!(GAUSSIAN, Gaussian::default());
    assert_eq!(EPANECHNIKOV, Epanechnikov::default());
    assert_eq!(TRIANGULAR, Triangular::default());
    assert_eq!(QUARTIC, Quartic::default());
    let t1 = Tricube;
    let t2 = t1;
    let t3 = t1.clone();
    assert_eq!(t1, t2);
    assert_eq!(t1, t3);
    let _ = format!("{:?}", TRICUBE);
}

#[test]
fn fn_pointer_implements_kernelfn() {
    let f: fn(f64) -> f64 = gaussian;
    assert!(approx_eq(f.weight(0.0), gaussian(0.0), 1e-15));
    assert!(approx_eq(
        (tricube as fn(f64) -> f64).weight(0.5),
        tricube(0.5),
        1e-15
    ));
}

#[test]
fn monotonicity_samples() {
    let kernels: [fn(f64) -> f64; 4] = [tricube, epanechnikov, quartic, triangular];
    let samples = [0.0_f64, 0.25, 0.5, 0.75, 0.99];
    for &k in &kernels {
        let mut prev = k(0.0);
        for &u in &samples[1..] {
            let cur = k(u);
            assert!(
                cur <= prev + 1e-12,
                "kernel not nonincreasing at u={}, prev={}, cur={}",
                u,
                prev,
                cur
            );
            prev = cur;
        }
    }
}

#[test]
fn integrate_tricube_to_one() {
    let integral = integrate(tricube, -1.0, 1.0, 10_000);
    let expected = 81.0 / 70.0;
    assert!(
        (integral - expected).abs() < 1e-3,
        "Tricube integral ≈ {}, expected ≈ {}",
        integral,
        expected
    );
}

#[test]
fn integrate_epanechnikov_to_one() {
    let integral = integrate(epanechnikov, -1.0, 1.0, 10_000);
    assert!(
        (integral - 1.0).abs() < 1e-3,
        "Epanechnikov integral ≈ {}",
        integral
    );
}

#[test]
fn integrate_quartic_to_one() {
    let integral = integrate(quartic, -1.0, 1.0, 10_000);
    assert!(
        (integral - 1.0).abs() < 1e-3,
        "Quartic integral ≈ {}",
        integral
    );
}

#[test]
fn integrate_triangular_to_one() {
    let integral = integrate(triangular, -1.0, 1.0, 10_000);
    assert!(
        (integral - 1.0).abs() < 1e-3,
        "Triangular integral ≈ {}",
        integral
    );
}
