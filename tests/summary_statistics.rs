use approx::{abs_diff_eq, assert_abs_diff_eq};
use ndarray::{arr0, array, Array, Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::{
    errors::{EmptyInput, MultiInputError, ShapeMismatch},
    SummaryStatisticsExt,
};
use noisy_float::types::N64;
use quickcheck::{quickcheck, TestResult};
use std::f64;

#[test]
fn test_with_nan_values() {
    let a = array![f64::NAN, 1.];
    let weights = array![1.0, f64::NAN];
    assert!(a.mean().unwrap().is_nan());
    assert!(a.weighted_mean(&weights).unwrap().is_nan());
    assert!(a.weighted_sum(&weights).unwrap().is_nan());
    assert!(a
        .weighted_mean_axis(Axis(0), &weights)
        .unwrap()
        .into_scalar()
        .is_nan());
    assert!(a
        .weighted_sum_axis(Axis(0), &weights)
        .unwrap()
        .into_scalar()
        .is_nan());
    assert!(a.harmonic_mean().unwrap().is_nan());
    assert!(a.geometric_mean().unwrap().is_nan());
    assert!(a.weighted_var(&weights, 0.0).unwrap().is_nan());
    assert!(a.weighted_std(&weights, 0.0).unwrap().is_nan());
    assert!(a
        .weighted_var_axis(Axis(0), &weights, 0.0)
        .unwrap()
        .into_scalar()
        .is_nan());
    assert!(a
        .weighted_std_axis(Axis(0), &weights, 0.0)
        .unwrap()
        .into_scalar()
        .is_nan());
}

#[test]
fn test_with_empty_array_of_floats() {
    let a: Array1<f64> = array![];
    let weights = array![1.0];
    assert_eq!(a.mean(), None);
    assert_eq!(a.weighted_mean(&weights), Err(MultiInputError::EmptyInput));
    assert_eq!(
        a.weighted_mean_axis(Axis(0), &weights),
        Err(MultiInputError::EmptyInput)
    );
    assert_eq!(a.harmonic_mean(), Err(EmptyInput));
    assert_eq!(a.geometric_mean(), Err(EmptyInput));
    assert_eq!(
        a.weighted_var(&weights, 0.0),
        Err(MultiInputError::EmptyInput)
    );
    assert_eq!(
        a.weighted_std(&weights, 0.0),
        Err(MultiInputError::EmptyInput)
    );
    assert_eq!(
        a.weighted_var_axis(Axis(0), &weights, 0.0),
        Err(MultiInputError::EmptyInput)
    );
    assert_eq!(
        a.weighted_std_axis(Axis(0), &weights, 0.0),
        Err(MultiInputError::EmptyInput)
    );

    // The sum methods accept empty arrays
    assert_eq!(a.weighted_sum(&array![]), Ok(0.0));
    assert_eq!(a.weighted_sum_axis(Axis(0), &array![]), Ok(arr0(0.0)));
}

#[test]
fn test_with_empty_array_of_noisy_floats() {
    let a: Array1<N64> = array![];
    let weights = array![];
    assert_eq!(a.mean(), None);
    assert_eq!(a.weighted_mean(&weights), Err(MultiInputError::EmptyInput));
    assert_eq!(
        a.weighted_mean_axis(Axis(0), &weights),
        Err(MultiInputError::EmptyInput)
    );
    assert_eq!(a.harmonic_mean(), Err(EmptyInput));
    assert_eq!(a.geometric_mean(), Err(EmptyInput));
    assert_eq!(
        a.weighted_var(&weights, N64::new(0.0)),
        Err(MultiInputError::EmptyInput)
    );
    assert_eq!(
        a.weighted_std(&weights, N64::new(0.0)),
        Err(MultiInputError::EmptyInput)
    );
    assert_eq!(
        a.weighted_var_axis(Axis(0), &weights, N64::new(0.0)),
        Err(MultiInputError::EmptyInput)
    );
    assert_eq!(
        a.weighted_std_axis(Axis(0), &weights, N64::new(0.0)),
        Err(MultiInputError::EmptyInput)
    );

    // The sum methods accept empty arrays
    assert_eq!(a.weighted_sum(&weights), Ok(N64::new(0.0)));
    assert_eq!(
        a.weighted_sum_axis(Axis(0), &weights),
        Ok(arr0(N64::new(0.0)))
    );
}

#[test]
fn test_with_array_of_floats() {
    let a: Array1<f64> = array![
        0.99889651, 0.0150731, 0.28492482, 0.83819218, 0.48413156, 0.80710412, 0.41762936,
        0.22879429, 0.43997224, 0.23831807, 0.02416466, 0.6269962, 0.47420614, 0.56275487,
        0.78995021, 0.16060581, 0.64635041, 0.34876609, 0.78543249, 0.19938356, 0.34429457,
        0.88072369, 0.17638164, 0.60819363, 0.250392, 0.69912532, 0.78855523, 0.79140914,
        0.85084218, 0.31839879, 0.63381769, 0.22421048, 0.70760302, 0.99216018, 0.80199153,
        0.19239188, 0.61356023, 0.31505352, 0.06120481, 0.66417377, 0.63608897, 0.84959691,
        0.43599069, 0.77867775, 0.88267754, 0.83003623, 0.67016118, 0.67547638, 0.65220036,
        0.68043427
    ];
    // Computed using NumPy
    let expected_mean = 0.5475494059146699;
    let expected_weighted_mean = 0.6782420496397121;
    let expected_weighted_var = 0.04306695637838332;
    // Computed using SciPy
    let expected_harmonic_mean = 0.21790094950226022;
    let expected_geometric_mean = 0.4345897639796527;

    assert_abs_diff_eq!(a.mean().unwrap(), expected_mean, epsilon = 1e-9);
    assert_abs_diff_eq!(
        a.harmonic_mean().unwrap(),
        expected_harmonic_mean,
        epsilon = 1e-7
    );
    assert_abs_diff_eq!(
        a.geometric_mean().unwrap(),
        expected_geometric_mean,
        epsilon = 1e-12
    );

    // Input array used as weights, normalized
    let weights = &a / a.sum();
    assert_abs_diff_eq!(
        a.weighted_sum(&weights).unwrap(),
        expected_weighted_mean,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        a.weighted_var(&weights, 0.0).unwrap(),
        expected_weighted_var,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        a.weighted_std(&weights, 0.0).unwrap(),
        expected_weighted_var.sqrt(),
        epsilon = 1e-12
    );

    let data = a.into_shape((2, 5, 5)).unwrap();
    let weights = array![0.1, 0.5, 0.25, 0.15, 0.2];
    assert_abs_diff_eq!(
        data.weighted_mean_axis(Axis(1), &weights).unwrap(),
        array![
            [0.50202721, 0.53347361, 0.29086033, 0.56995637, 0.37087139],
            [0.58028328, 0.50485216, 0.59349973, 0.70308937, 0.72280630]
        ],
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        data.weighted_mean_axis(Axis(2), &weights).unwrap(),
        array![
            [0.33434378, 0.38365259, 0.56405781, 0.48676574, 0.55016179],
            [0.71112376, 0.55134174, 0.45566513, 0.74228516, 0.68405851]
        ],
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        data.weighted_sum_axis(Axis(1), &weights).unwrap(),
        array![
            [0.60243266, 0.64016833, 0.34903240, 0.68394765, 0.44504567],
            [0.69633993, 0.60582259, 0.71219968, 0.84370724, 0.86736757]
        ],
        epsilon = 1e-8
    );
    assert_abs_diff_eq!(
        data.weighted_sum_axis(Axis(2), &weights).unwrap(),
        array![
            [0.40121254, 0.46038311, 0.67686937, 0.58411889, 0.66019415],
            [0.85334851, 0.66161009, 0.54679815, 0.89074219, 0.82087021]
        ],
        epsilon = 1e-8
    );
}

#[test]
fn weighted_sum_dimension_zero() {
    let a = Array2::<usize>::zeros((0, 20));
    assert_eq!(
        a.weighted_sum_axis(Axis(0), &Array1::zeros(0)).unwrap(),
        Array1::from_elem(20, 0)
    );
    assert_eq!(
        a.weighted_sum_axis(Axis(1), &Array1::zeros(20)).unwrap(),
        Array1::from_elem(0, 0)
    );
    assert_eq!(
        a.weighted_sum_axis(Axis(0), &Array1::zeros(1)),
        Err(MultiInputError::ShapeMismatch(ShapeMismatch {
            first_shape: vec![0, 20],
            second_shape: vec![1]
        }))
    );
    assert_eq!(
        a.weighted_sum(&Array2::zeros((10, 20))),
        Err(MultiInputError::ShapeMismatch(ShapeMismatch {
            first_shape: vec![0, 20],
            second_shape: vec![10, 20]
        }))
    );
}

#[test]
fn mean_eq_if_uniform_weights() {
    fn prop(a: Vec<f64>) -> TestResult {
        if a.len() < 1 {
            return TestResult::discard();
        }
        let a = Array1::from(a);
        let weights = Array1::from_elem(a.len(), 1.0 / a.len() as f64);
        let m = a.mean().unwrap();
        let wm = a.weighted_mean(&weights).unwrap();
        let ws = a.weighted_sum(&weights).unwrap();
        TestResult::from_bool(
            abs_diff_eq!(m, wm, epsilon = 1e-9) && abs_diff_eq!(wm, ws, epsilon = 1e-9),
        )
    }
    quickcheck(prop as fn(Vec<f64>) -> TestResult);
}

#[test]
fn mean_axis_eq_if_uniform_weights() {
    fn prop(mut a: Vec<f64>) -> TestResult {
        if a.len() < 24 {
            return TestResult::discard();
        }
        let depth = a.len() / 12;
        a.truncate(depth * 3 * 4);
        let weights = Array1::from_elem(depth, 1.0 / depth as f64);
        let a = Array1::from(a).into_shape((depth, 3, 4)).unwrap();
        let ma = a.mean_axis(Axis(0)).unwrap();
        let wm = a.weighted_mean_axis(Axis(0), &weights).unwrap();
        let ws = a.weighted_sum_axis(Axis(0), &weights).unwrap();
        TestResult::from_bool(
            abs_diff_eq!(ma, wm, epsilon = 1e-12) && abs_diff_eq!(wm, ws, epsilon = 1e12),
        )
    }
    quickcheck(prop as fn(Vec<f64>) -> TestResult);
}

#[test]
fn weighted_var_eq_var_if_uniform_weight() {
    fn prop(a: Vec<f64>) -> TestResult {
        if a.len() < 1 {
            return TestResult::discard();
        }
        let a = Array1::from(a);
        let weights = Array1::from_elem(a.len(), 1.0 / a.len() as f64);
        let weighted_var = a.weighted_var(&weights, 0.0).unwrap();
        let var = a.var_axis(Axis(0), 0.0).into_scalar();
        TestResult::from_bool(abs_diff_eq!(weighted_var, var, epsilon = 1e-10))
    }
    quickcheck(prop as fn(Vec<f64>) -> TestResult);
}

#[test]
fn weighted_var_algo_eq_simple_algo() {
    fn prop(mut a: Vec<f64>) -> TestResult {
        if a.len() < 24 {
            return TestResult::discard();
        }
        let depth = a.len() / 12;
        a.truncate(depth * 3 * 4);
        let a = Array1::from(a).into_shape((depth, 3, 4)).unwrap();
        let mut success = true;
        for axis in 0..3 {
            let axis = Axis(axis);

            let weights = Array::random(a.len_of(axis), Uniform::new(0.0, 1.0));
            let mean = a
                .weighted_mean_axis(axis, &weights)
                .unwrap()
                .insert_axis(axis);
            let res_1_pass = a.weighted_var_axis(axis, &weights, 0.0).unwrap();
            let res_2_pass = (&a - &mean)
                .mapv_into(|v| v.powi(2))
                .weighted_mean_axis(axis, &weights)
                .unwrap();
            success &= abs_diff_eq!(res_1_pass, res_2_pass, epsilon = 1e-10);
        }
        TestResult::from_bool(success)
    }
    quickcheck(prop as fn(Vec<f64>) -> TestResult);
}

#[test]
fn test_central_moment_with_empty_array_of_floats() {
    let a: Array1<f64> = array![];
    for order in 0..=3 {
        assert_eq!(a.central_moment(order), Err(EmptyInput));
        assert_eq!(a.central_moments(order), Err(EmptyInput));
    }
}

#[test]
fn test_zeroth_central_moment_is_one() {
    let n = 50;
    let bound: f64 = 200.;
    let a = Array::random(n, Uniform::new(-bound.abs(), bound.abs()));
    assert_eq!(a.central_moment(0).unwrap(), 1.);
}

#[test]
fn test_first_central_moment_is_zero() {
    let n = 50;
    let bound: f64 = 200.;
    let a = Array::random(n, Uniform::new(-bound.abs(), bound.abs()));
    assert_eq!(a.central_moment(1).unwrap(), 0.);
}

#[test]
fn test_central_moments() {
    let a: Array1<f64> = array![
        0.07820559, 0.5026185, 0.80935324, 0.39384033, 0.9483038, 0.62516215, 0.90772261,
        0.87329831, 0.60267392, 0.2960298, 0.02810356, 0.31911966, 0.86705506, 0.96884832,
        0.2222465, 0.42162446, 0.99909868, 0.47619762, 0.91696979, 0.9972741, 0.09891734,
        0.76934818, 0.77566862, 0.7692585, 0.2235759, 0.44821286, 0.79732186, 0.04804275,
        0.87863238, 0.1111003, 0.6653943, 0.44386445, 0.2133176, 0.39397086, 0.4374617, 0.95896624,
        0.57850146, 0.29301706, 0.02329879, 0.2123203, 0.62005503, 0.996492, 0.5342986, 0.97822099,
        0.5028445, 0.6693834, 0.14256682, 0.52724704, 0.73482372, 0.1809703,
    ];
    // Computed using scipy.stats.moment
    let expected_moments = vec![
        1.,
        0.,
        0.09339920262960291,
        -0.0026849636727735186,
        0.015403769257729755,
        -0.001204176487006564,
        0.002976822584939186,
    ];
    for (order, expected_moment) in expected_moments.iter().enumerate() {
        assert_abs_diff_eq!(
            a.central_moment(order as u16).unwrap(),
            expected_moment,
            epsilon = 1e-8
        );
    }
}

#[test]
fn test_bulk_central_moments() {
    // Test that the bulk method is coherent with the non-bulk method
    let n = 50;
    let bound: f64 = 200.;
    let a = Array::random(n, Uniform::new(-bound.abs(), bound.abs()));
    let order = 10;
    let central_moments = a.central_moments(order).unwrap();
    for i in 0..=order {
        assert_eq!(a.central_moment(i).unwrap(), central_moments[i as usize]);
    }
}

#[test]
fn test_kurtosis_and_skewness_is_none_with_empty_array_of_floats() {
    let a: Array1<f64> = array![];
    assert_eq!(a.skewness(), Err(EmptyInput));
    assert_eq!(a.kurtosis(), Err(EmptyInput));
}

#[test]
fn test_kurtosis_and_skewness() {
    let a: Array1<f64> = array![
        0.33310096, 0.98757449, 0.9789796, 0.96738114, 0.43545674, 0.06746873, 0.23706562,
        0.04241815, 0.38961714, 0.52421271, 0.93430327, 0.33911604, 0.05112372, 0.5013455,
        0.05291507, 0.62511183, 0.20749633, 0.22132433, 0.14734804, 0.51960608, 0.00449208,
        0.4093339, 0.2237519, 0.28070469, 0.7887231, 0.92224523, 0.43454188, 0.18335111,
        0.08646856, 0.87979847, 0.25483457, 0.99975627, 0.52712442, 0.41163279, 0.85162594,
        0.52618733, 0.75815023, 0.30640695, 0.14205781, 0.59695813, 0.851331, 0.39524328,
        0.73965373, 0.4007615, 0.02133069, 0.92899207, 0.79878191, 0.38947334, 0.22042183,
        0.77768353,
    ];
    // Computed using scipy.stats.kurtosis(a, fisher=False)
    let expected_kurtosis = 1.821933711687523;
    // Computed using scipy.stats.skew
    let expected_skewness = 0.2604785422878771;

    let kurtosis = a.kurtosis().unwrap();
    let skewness = a.skewness().unwrap();

    assert_abs_diff_eq!(kurtosis, expected_kurtosis, epsilon = 1e-12);
    assert_abs_diff_eq!(skewness, expected_skewness, epsilon = 1e-8);
}
