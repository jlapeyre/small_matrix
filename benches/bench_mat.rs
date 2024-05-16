use criterion::{black_box, criterion_group, criterion_main, Criterion};

use small_matrix_rust::{
    matmul_2x2_row_major,
    matmul_2x2_row_major_unrolled,
    matmul_4x4_row_major,
    matmul_4x4_col_major,
    matmul_4x4_faer_row_major,
    matmul_4x4_faer_col_major,
    matmul_4x4_array_view_row_major,
    matmul_4x4_array_view_col_major,
    matmul_nxn_array_view_col_major,
    matmul_nxn_array_view_row_major,
    matmul_nxn_row_major,
    matmul_nxn_col_major,
//    matmul_nxn_array_view,
//    matmul_4x4_checked,
};

use num_complex::{Complex64, Complex};
use ndarray::{Array2, aview2, aview_mut2};
use ndarray::linalg::general_mat_mul;

use faer::prelude::*;
use faer::{Mat, Parallelism};
use faer::modules::core::mul::matmul;

fn _ndarray_get_complex() -> [Array2<Complex64>; 3]  {
    let c = Array2::<Complex64>::default((4, 4));
    let a = Array2::<Complex64>::default((4, 4));
    let b = Array2::<Complex64>::default((4, 4));
    return [c, a, b]
}

fn _ndarray_get_real() -> [Array2<f64>; 3]  {
    let c = Array2::<f64>::default((4, 4));
    let a = Array2::<f64>::default((4, 4));
    let b = Array2::<f64>::default((4, 4));
    return [c, a, b]
}

fn _rust_array_get_complex() -> [[[Complex<f64>; 4]; 4]; 3]  {
    let cz = Complex64::new(0., 0.);
    let c1 = Complex64::new(1., 0.);
    let c: [[Complex64; 4]; 4] = [[cz, cz, cz, cz,], [cz, cz, cz, cz,], [cz, cz, cz, cz,], [cz, cz, cz, cz,]];
    let a: [[Complex64; 4]; 4]  = [[c1, c1, c1, c1,], [c1, c1, c1, c1,], [c1, c1, c1, c1,], [c1, c1, c1, c1,]];
    let b: [[Complex64; 4]; 4]  = [[c1, c1, c1, c1,], [c1, c1, c1, c1,], [c1, c1, c1, c1,], [c1, c1, c1, c1,]];
    return [c, a, b]
}

// fn _rust_array_get_complex() -> [[[Complex<f64>; 4]; 4]; 3]  {
// }

fn mixed_array_complex_col(crit: &mut Criterion) {
    let [mut c, a, b] = _rust_array_get_complex();
    let a1 = aview2(&a);
    let b1 = aview2(&b);
    crit.bench_function("mixed_array_cplx_generic_4x4_col",
                        |bench| bench.iter(|| matmul_4x4_col_major(black_box(&mut c), black_box(a1), black_box(b1))));
}

fn rust_array_complex_col(crit: &mut Criterion) {
    let [mut c, a, b] = _rust_array_get_complex();
    crit.bench_function("rust_array_cplx_generic_4x4_col",
                        |bench| bench.iter(|| matmul_4x4_col_major(black_box(&mut c), black_box(&a), black_box(&b))));
}

fn rust_array_complex_row(crit: &mut Criterion) {
    let [mut c, a, b] = _rust_array_get_complex();
    crit.bench_function("rust_array_cplx_generic_4x4_row",
                        |bench| bench.iter(|| matmul_4x4_row_major(black_box(&mut c), black_box(&a), black_box(&b))));
}

fn rust_array_2x2_complex_row(crit: &mut Criterion) {

}

fn ndarray_2x2_complex_row(crit: &mut Criterion) {
    let _a = [[Complex64::new(1., 0.); 2]; 2];
    let _b = [[Complex64::new(1., 0.); 2]; 2];
    let mut _c = [[Complex64::new(1., 0.); 2]; 2];
    let a = aview2(&_a).to_owned();
    let b = aview2(&_b).to_owned();
    let mut c = aview_mut2(&mut _c).to_owned();
    crit.bench_function("ndarray_cplx_generic_2x2_row",
                        |bench| bench.iter(|| matmul_2x2_row_major(black_box(c.view_mut()), black_box(a.view()), black_box(b.view()))));
}

fn ndarray_2x2_complex_row_unrolled(crit: &mut Criterion) {
    let _a = [[Complex64::new(1., 0.); 2]; 2];
    let _b = [[Complex64::new(1., 0.); 2]; 2];
    let mut _c = [[Complex64::new(1., 0.); 2]; 2];
    let a = aview2(&_a).to_owned();
    let b = aview2(&_b).to_owned();
    let mut c = aview_mut2(&mut _c).to_owned();
    crit.bench_function("ndarray_cplx_generic_2x2_row_unrolled",
                        |bench| bench.iter(|| matmul_2x2_row_major_unrolled(black_box(c.view_mut()), black_box(a.view()), black_box(b.view()))));
}

fn ndarray_complex_row(crit: &mut Criterion) {
    let [mut c, a, b] = _ndarray_get_complex();
    crit.bench_function("ndarray_cplx_generic_4x4_row",
                        |bench| bench.iter(|| matmul_4x4_row_major(black_box(c.view_mut()), black_box(a.view()), black_box(b.view()))));
}

fn ndarray_complex_col(crit: &mut Criterion) {
    let [mut c, a, b] = _ndarray_get_complex();
    crit.bench_function("ndarray_cplx_generic_4x4_col",
                        |bench| bench.iter(|| matmul_4x4_col_major(black_box(c.view_mut()), black_box(a.view()), black_box(b.view()))));
}

// This is ok. But we have too many tests
// fn ndarray_complex_checked(crit: &mut Criterion) {
//     let [mut c, a, b] = _ndarray_get_complex();
//     crit.bench_function("ndarray_cplx_generic_4x4_checked",
//                         |bench| bench.iter(|| matmul_4x4_checked(black_box(c.view_mut()), black_box(a.view()), black_box(b.view()))));
// }

fn ndarray_complex_array_view_row(crit: &mut Criterion) {
    let [mut c, a, b] = _ndarray_get_complex();
    crit.bench_function("ndarray_cplx_4x4_native_index_row",
                        |bench| bench.iter(|| matmul_4x4_array_view_row_major(black_box(c.view_mut()), black_box(a.view()), black_box(b.view()))));
}

fn ndarray_complex_array_view_col(crit: &mut Criterion) {
    let [mut c, a, b] = _ndarray_get_complex();
    crit.bench_function("ndarray_cplx_4x4_native_index_col",
                        |bench| bench.iter(|| matmul_4x4_array_view_col_major(black_box(c.view_mut()), black_box(a.view()), black_box(b.view()))));
}

fn ndarray_complex_array_view_nxn_row(crit: &mut Criterion) {
    let [mut c, a, b] = _ndarray_get_complex();
    crit.bench_function("ndarray_cplx_nxn_native_index_row",
                        |bench| bench.iter(|| matmul_nxn_array_view_row_major(black_box(c.view_mut()), black_box(a.view()), black_box(b.view()))));
}

// ndarray-native, non-allocating matrix multiplication
fn ndarray_complex_general_mat_mul(crit: &mut Criterion) {
    let [mut c, a, b] = _ndarray_get_complex();
    crit.bench_function("ndarray_cplx_native",
                        |bench| bench
                        .iter(|| general_mat_mul(black_box(Complex64::new(1.0, 0.0)),
                                                 black_box(&a.view()), black_box(&b.view()),
                                                 black_box(Complex64::new(0.0, 0.0)), black_box(&mut c.view_mut()))));
}

fn ndarray_complex_nxn_row(crit: &mut Criterion) {
    let [mut c, a, b] = _ndarray_get_complex();
    crit.bench_function("ndarray_cplx_generic_nxn_row",
                        |bench| bench.iter(|| matmul_nxn_row_major(black_box(c.view_mut()), black_box(a.view()), black_box(b.view()))));
}

fn _faer_get_complex() -> [Mat<c64>; 3]{
    let c = Mat::<c64>::identity(4, 4);
    let a = Mat::<c64>::zeros(4, 4);
    let b = Mat::<c64>::zeros(4, 4);
    [c, a, b]
}

fn _faer_get_real() -> [Mat<f64>; 3]{
    let c = Mat::<f64>::identity(4, 4);
    let a = Mat::<f64>::zeros(4, 4);
    let b = Mat::<f64>::zeros(4, 4);
    [c, a, b]
}

// faer Mat using our most generic routine.
fn faer_complex_row(crit: &mut Criterion) {
    let [mut c, a, b] = _faer_get_complex();
    crit.bench_function("faer_cplx_generic_4x4_row",
                        |bench| bench.iter(|| matmul_4x4_row_major(black_box(c.as_mut()), black_box(a.as_ref()), black_box(b.as_ref()))));
}

// faer Mat using our most generic routine.
fn faer_complex_col(crit: &mut Criterion) {
    let [mut c, a, b] = _faer_get_complex();
    crit.bench_function("faer_cplx_generic_4x4_col",
                        |bench| bench.iter(|| matmul_4x4_col_major(black_box(c.as_mut()), black_box(a.as_ref()), black_box(b.as_ref()))));
}


// // faer Mat using our most generic_4x4 routine.
// fn faer_complex_checked(crit: &mut Criterion) {
//     let [mut c, a, b] = _faer_get_complex();
//     crit.bench_function("faer_cplx_generic_4x4_checked",
//                         |bench| bench.iter(|| matmul_4x4_checked(black_box(c.as_mut()), black_box(a.as_ref()), black_box(b.as_ref()))));
// }

// faer Mat using our most generic routine.
// fn faer_complex_nxn_row(crit: &mut Criterion) {
//     let [mut c, a, b] = _faer_get_complex();
//     crit.bench_function("faer_cplx_generic_nxn",
//                         |bench| bench.iter(|| matmul_nxn_row(black_box(c.as_mut()), black_box(a.as_ref()), black_box(b.as_ref()))));
// }

// faer-native indexing with naive, loops.
fn faer_complex_native_index_row(crit: &mut Criterion) {
    let [mut c, a, b] = _faer_get_complex();
    crit.bench_function("faer_cplx_4x4_native_index_row",
                        |bench| bench.iter(|| matmul_4x4_faer_row_major(black_box(c.as_mut()), black_box(a.as_ref()), black_box(b.as_ref()))));
}

// faer-native indexing with naive, loops.
fn faer_complex_native_index_col(crit: &mut Criterion) {
    let [mut c, a, b] = _faer_get_complex();
    crit.bench_function("faer_cplx_4x4_native_index_col",
                        |bench| bench.iter(|| matmul_4x4_faer_col_major(black_box(c.as_mut()), black_box(a.as_ref()), black_box(b.as_ref()))));
}

// // faer-native indexing with naive, loops.
// fn faer_complex_only_native_index(crit: &mut Criterion) {
//     let [mut c, a, b] = _faer_get_complex();
//     crit.bench_function("faer_cplx_only_4x4_native_index",
//                         |bench| bench.iter(|| matmul_4x4_faer_complex(black_box(c.as_mut()), black_box(a.as_ref()), black_box(b.as_ref()))));
// }

// faer-native, non-allocating matrix multiplication
fn faer_complex_matmul(crit: &mut Criterion) {
//    let [mut c, a, b] = _faer_get_complex();
    let mut c = Mat::<c64>::identity(4, 4);
    let a = Mat::<c64>::zeros(4, 4);
    let b = Mat::<c64>::zeros(4, 4);
    crit.bench_function("faer_cplx_native",
                        |bench| bench
                        .iter(||
        matmul(
            c.as_mut(),
            a.as_ref(),
            b.as_ref(),
            None,
            c64::new(1., 0.),
            Parallelism::None,
        )));
}

fn ndarray_real_row(crit: &mut Criterion) {
    let [mut c, a, b] = _ndarray_get_real();
    crit.bench_function("ndarray_real_generic_4x4_row",
                        |bench| bench.iter(|| matmul_4x4_row_major(black_box(c.view_mut()), black_box(a.view()), black_box(b.view()))));
}

fn ndarray_real_array_view_row(crit: &mut Criterion) {
    let [mut c, a, b] = _ndarray_get_real();
    crit.bench_function("ndarray_real_generic_4x4_native_index_row",
                        |bench| bench.iter(|| matmul_4x4_array_view_row_major(black_box(c.view_mut()), black_box(a.view()), black_box(b.view()))));
}

fn faer_real_row(crit: &mut Criterion) {
    let [mut c, a, b] = _faer_get_real();
    crit.bench_function("faer_real_generic_4x4_row",
                        |bench| bench.iter(|| matmul_4x4_row_major(black_box(c.as_mut()), black_box(a.as_ref()), black_box(b.as_ref()))));
}

fn faer_real_mat_row(crit: &mut Criterion) {
    let [mut c, a, b] = _faer_get_real();
    crit.bench_function("faer_real_generic_4x4_native_index_row",
                        |bench| bench.iter(|| matmul_4x4_faer_row_major(black_box(c.as_mut()), black_box(a.as_ref()), black_box(b.as_ref()))));
}

// This order has no effect on the order of the html page of results. Looks like the latter may be sorted
// alphabetically.
// criterion_group!(benches,
//                  ndarray_complex,
//                  ndarray_complex_array_view,
// );

criterion_group!(benches,
                 // rust_array_2x2_complex_row,
                 // ndarray_2x2_complex_row,
                 // ndarray_2x2_complex_row_unrolled,
                 mixed_array_complex_col,
                 rust_array_complex_col,
                 rust_array_complex_row,
                 ndarray_complex_row,
                 ndarray_complex_col,
                 ndarray_complex_array_view_row,
                 ndarray_complex_array_view_col,
                 ndarray_complex_general_mat_mul,
                 faer_complex_row,
                 faer_complex_col,
                 faer_complex_native_index_row,
                 faer_complex_native_index_col,
                 faer_complex_matmul,
                 // faer_real,
                 // faer_real_mat,
                 // ndarray_real,
                 // ndarray_real_array_view,
                 //                 ndarray_complex_array_view_nxn,
                 //                 ndarray_complex_array_view_nxn,
                 //faer_complex_nxn,
);


// The following have been checked a few times.
// They slow benchmarking and clutter the results list
// ndarray_real_literal, ndarray_complex_literal
criterion_main!(benches);
