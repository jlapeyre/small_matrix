use num_complex::{Complex64};
use ndarray::{aview2, arr2, Array2, ArrayView2};
use ndarray;

// ZERO and ONE are defined in num_complex 0.4.6
const ZERO: Complex64 = Complex64::new(0., 0.);
const ONE: Complex64 = Complex64::new(1., 0.);
const M_ONE: Complex64 = Complex64::new(-1., 0.);
// const IM: Complex64 = Complex64::new(0., 1.);
// const M_IM: Complex64 = Complex64::new(0., -1.);

pub static CZGATE: [[Complex64; 4]; 4] = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ONE, ZERO],
    [ZERO, ZERO, ZERO, M_ONE],
];

pub fn czgate() -> Array2<Complex64> {
    aview2(&CZGATE).to_owned()
}

pub fn czgate4() -> Array2<Complex64> {
    arr2(&CZGATE)
}

// macro_rules! marray {
//     ($name: ident, $fname: ident, $y: tt) => {
//         pub static $name: [[Complex64; 4]; 4] = $y;
//         pub fn $fname() -> ArrayView2<'static, Complex64> {
//             aview2(&$name)
//         }
//     };
// }

macro_rules! gate_1q {
    ($gate_name: ident, $func_name: ident,
     [[$m00: expr , $m01: expr] , [ $m10: expr, $m11: expr ] $(,)*]) => {
        pub static $gate_name: [[Complex64; 2]; 2] = [
            [$m00, $m01],
            [$m10, $m11]
            ];
        pub fn $func_name() -> ArrayView2<'static, Complex64> {
            aview2(&$gate_name)
        }
    };
}

// macro_rules! gate_1qold {
//     ($gate_name: ident, $func_name: ident, $y: tt) => {
//         pub static $gate_name: [[Complex64; 2]; 2] = $y;
//         pub fn $func_name() -> ArrayView2<'static, Complex64> {
//             aview2(&$gate_name)
//         }
//     };
// }

macro_rules! gate_2q {
//    ($gate_name: ident, $func_name: ident, $y: tt) => {
    ($gate_name: ident, $func_name: ident, $y: tt) => {
        pub static $name: [[Complex64; 4]; 4] = $y;
        pub fn $fname() -> ArrayView2<'static, Complex64> {
            aview2(&$name)
        }
    };
}

// marray! [AGATE, a_gate, [
//     [ONE, ZERO, ZERO, ZERO],
//     [ZERO, ONE, ZERO, ZERO],
//     [ZERO, ZERO, ONE, ZERO],
//     [ZERO, ZERO, ZERO, M_ONE]
// ]];


gate_1q!(OTHERGATE, other_gate, [
    [ZERO, ZERO],
    [ONE, ONE],
]);

// gate_1q!(OTHERGATE1, other_gate1, [
//     [ZERO, ZERO],
//     [ONE, ONE],
//     [ONE, ONE]
// ]);


pub fn trytwo() -> Array2<Complex64> {
    Array2::from(vec![[ONE, ONE], [ONE, ONE]])
//    Array2::from(vec![[ONE, ONE, ONE, ONE]])
}

macro_rules! testmac {
    ($x: expr) =>  { $x };
}


//pub fn czgate3() -> ArrayBase<ViewRepr<&Complex<f64>>, Dim<[usize; 2]>> {
pub fn czgate3() -> ArrayView2<'static, Complex64> {
    testmac!(1);
    testmac!(aview2(&CZGATE))
}
