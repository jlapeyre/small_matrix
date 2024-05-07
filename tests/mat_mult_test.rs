use small_matrix_rust::num_complex::{c64, ZERO, CMatrix4x4};
use small_matrix_rust::matmul::{
//    matmul_4x4_row_major,
    matmul_4x4_col_major,
//    matmul_nxn_col_major,
};

use num_complex::{Complex64};
use ndarray::{arr2, Array2};

use faer_ext::IntoFaerComplex;


// We choose two matrices to multiply that don't have much symmetry.
// We can check the matrix multiplication with Julia:
//     struct c64 a; b end;  # This is to display like Rust input
//     c64(z::Complex) = c64(real(z), imag(z));
//     a = reshape([ComplexF64(i - 1, i+j - 2) for j in 1:4 for i in 1:4], (4,4));
//     b = reshape([ComplexF64(i, i+j - 1) for j in 1:4 for i in 1:4], (4,4));
//     c64.(a * b)
//
// 4×4 Matrix{c64}:
//  c64(-20.0, 20.0)  c64(-26.0, 20.0)  c64(-32.0, 20.0)   c64(-38.0, 20.0)
//  c64(-20.0, 40.0)  c64(-30.0, 44.0)  c64(-40.0, 48.0)   c64(-50.0, 52.0)
//  c64(-20.0, 60.0)  c64(-34.0, 68.0)  c64(-48.0, 76.0)   c64(-62.0, 84.0)
//  c64(-20.0, 80.0)  c64(-38.0, 92.0)  c64(-56.0, 104.0)  c64(-74.0, 116.0)

fn result() -> CMatrix4x4 {
    [
        [c64(-20.0, 20.0),  c64(-26.0, 20.0),  c64(-32.0, 20.0),   c64(-38.0, 20.0),],
        [c64(-20.0, 40.0),  c64(-30.0, 44.0),  c64(-40.0, 48.0),   c64(-50.0, 52.0),],
        [c64(-20.0, 60.0),  c64(-34.0, 68.0),  c64(-48.0, 76.0),   c64(-62.0, 84.0),],
        [c64(-20.0, 80.0),  c64(-38.0, 92.0),  c64(-56.0, 104.0),  c64(-74.0, 116.0),],
    ]
}

fn array_a() -> CMatrix4x4 {
    let mut a = [[ZERO; 4]; 4];
    let ri: Vec<_> = (0..4).map(|x| x as i32).collect();
    let rj = ri.clone();
    for j in &rj {
        for i in &ri {
            a[*i as usize][*j as usize] = c64(*i, *i + *j);
        }
    }
    a
}

fn array_b() -> CMatrix4x4 {
    let mut b = [[ZERO; 4]; 4];
    let ri: Vec<_> = (0..4).map(|x| x as i32).collect();
    let rj = ri.clone();
    for j in &rj {
        for i in &ri {
            b[*i as usize][*j as usize] = c64(*i + 1, *i + *j + 1);
        }
    }
    b
}

fn get_4x4_rust_arrays() -> [CMatrix4x4; 3] {
    [array_a(), array_b(), [[ZERO; 4]; 4]]
}

fn get_4x4_ndarrays() -> [Array2<Complex64>; 3] {
    [arr2(&array_a()), arr2(&array_b()), arr2(&[[ZERO; 4]; 4])]
}

#[test]
fn test_rust_array_4x4() {
    let [a, b, mut c] = get_4x4_rust_arrays();
    matmul_4x4_col_major(&mut c, &a, &b);
    assert_eq!(c, result())
}

#[test]
fn test_ndarray_4x4() {
    let [a, b, mut c] = get_4x4_ndarrays();
    matmul_4x4_col_major(c.view_mut(), a.view(), b.view());
    assert_eq!(c, arr2(&result()))
}

#[test]
fn test_faer_4x4() {
    let [a, b, mut c] = get_4x4_ndarrays();
    let af = a.view().into_faer_complex();
    let bf = b.view().into_faer_complex();
    let mut cf = c.view_mut().into_faer_complex();
    matmul_4x4_col_major(cf.as_mut(), af.as_ref(), bf.as_ref());
    assert_eq!(&cf, &arr2(&result()).view().into_faer_complex());
}

#[test]
fn test_mixed_array_4x4() {
    let [a, _b, mut c] = get_4x4_rust_arrays();
    let [_a1, b1, _c1] = get_4x4_ndarrays();
    matmul_4x4_col_major(&mut c, &a, b1.view());
    assert_eq!(c, result())
}
