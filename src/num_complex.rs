use num_complex::{Complex64, Complex};

pub const ZERO: Complex64  = Complex64::new(0., 0.);
pub const ONE: Complex64 = Complex64::new(1., 0.);
pub const IM: Complex64 = Complex64::new(0., 1.);
pub type CMatrix4x4 = [[Complex64; 4]; 4];
pub type CMatrix2x2 = [[Complex64; 2]; 2];

// This is almost the same as the function that became available in
// num-complex 0.4.6. The difference is that two generic parameters are
// used here rather than one. This allows call like `c64(half_theta.cos(), 0);`
// that mix f64 and integer arguments.
/// Create a new [`Complex<f64>`] with arguments that can convert [`Into<f64>`].
///
/// ```
/// use num_complex::{c64, Complex64};
/// assert_eq!(c64(1, 2.), Complex64::new(1.0, 2.0));
/// ```
#[inline]
pub fn c64<T: Into<f64>, V: Into<f64>>(re: T, im: V) -> Complex64 {
    Complex::new(re.into(), im.into())
}
