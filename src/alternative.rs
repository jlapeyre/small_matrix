use ndarray::{ArrayView2, ArrayViewMut2};

use faer::prelude::*;
use faer_entity::IdentityGroup;
use faer::{Entity};

use crate::matmul::{MyZero};

// These are alternative implementations used for testing and benchmarkin comparison
// to the preferred implementations.

// Difference from matmul_4x4 is indexing methods from ndarray library rather than
// the traits defined above.
pub fn matmul_4x4_array_view_row_major<T>(mut c: ArrayViewMut2<T>, a: ArrayView2<T>, b: ArrayView2<T>)
where T:  std::ops::Add<Output = T>  + std::ops::Mul<Output = T> + std::ops::AddAssign + Default + Copy
{
    let (r0, r1) = (0, 4);
    for j in r0..r1 {
        for i in r0..r1 {
            // rely on this being additive id.
            let mut s = T::default();
            for k in r0..r1 {
                s += a[[i, k]] * b[[k, j]];
            }
            c[[i, j]] = s;
        }
    }
}

pub fn matmul_4x4_array_view_col_major<T>(mut c: ArrayViewMut2<T>, a: ArrayView2<T>, b: ArrayView2<T>)
where T:  std::ops::Add<Output = T>  + std::ops::Mul<Output = T> + std::ops::AddAssign + Default + Copy
{
    let (r0, r1) = (0, 4);
    for i in r0..r1 {
        for j in r0..r1 {
            // rely on this being additive id.
            let mut s = T::default();
            for k in r0..r1 {
                s += a[[i, k]] * b[[k, j]];
            }
            c[[i, j]] = s;
        }
    }
}

// pub fn matmul_nxn_array_view<T>(mut c: ArrayViewMut2<T>, a: ArrayView2<T>, b: ArrayView2<T>)
// where T:  std::ops::Add<Output = T>  + std::ops::Mul<Output = T> + std::ops::AddAssign + Default + Copy
// {
//     let n = a.ncols();
//     for i in 0..n {
//         for j in 0..n {
//             // rely on this being additive id.
//             let mut s = T::default();
//             for k in 0..n {
//                 s += a[[i, k]] * b[[k, j]];
//             }
//             c[[i, j]] = s;
//         }
//     }
// }

pub fn matmul_nxn_array_view_row_major<T>(mut c: ArrayViewMut2<T>, a: ArrayView2<T>, b: ArrayView2<T>)
where T:  std::ops::Add<Output = T>  + std::ops::Mul<Output = T> + std::ops::AddAssign + Default + Copy
{
    let n = a.ncols();
    for j in 0..n {
        for i in 0..n {
            // rely on this being additive id.
            let mut s = T::default();
            for k in 0..n {
                s += a[[i, k]] * b[[k, j]];
            }
            c[[i, j]] = s;
        }
    }
}

pub fn matmul_nxn_array_view_col_major<T>(mut c: ArrayViewMut2<T>, a: ArrayView2<T>, b: ArrayView2<T>)
where T:  std::ops::Add<Output = T>  + std::ops::Mul<Output = T> + std::ops::AddAssign + Default + Copy
{
    let n = a.ncols();
    for i in 0..n {
        for j in 0..n {
            // rely on this being additive id.
            let mut s = T::default();
            for k in 0..n {
                s += a[[i, k]] * b[[k, j]];
            }
            c[[i, j]] = s;
        }
    }
}

// Use indexing methods from faer library rather than those defined above.
// In principle, the generic routine use zero-cost abstractions to obtain
// the same performance.
// This method with explicit-ish indexing may be used to verify that there are no
// inefficiencies in the generic version.
pub fn matmul_4x4_faer_row_major<T>(mut c: MatMut<T>, a: MatRef<T>, b: MatRef<T>)
where T:  std::ops::Add<Output = T>  + std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T>  + Copy
    + Entity<Group = IdentityGroup, Unit=T>
{
    let (r0, r1) = (0, 4);
    for j in r0..r1 {
        for i in r0..r1 {
            // rely on this being additive id.
            let mut s = T::my_zero();
            for k in r0..r1 {
                s += a[(i, k)] * b[(k, j)];
            }
            c[(i, j)] = s;
        }
    }
}

// Use indexing methods from faer library rather than those defined above.
// In principle, the generic routine use zero-cost abstractions to obtain
// the same performance.
// This method with explicit-ish indexing may be used to verify that there are no
// inefficiencies in the generic version.
pub fn matmul_4x4_faer_col_major<T>(mut c: MatMut<T>, a: MatRef<T>, b: MatRef<T>)
where T:  std::ops::Add<Output = T>  + std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T>  + Copy
    + Entity<Group = IdentityGroup, Unit=T>
{
    let (r0, r1) = (0, 4);
    for i in r0..r1 {
        for j in r0..r1 {
            // rely on this being additive id.
            let mut s = T::my_zero();
            for k in r0..r1 {
                s += a[(i, k)] * b[(k, j)];
            }
            c[(i, j)] = s;
        }
    }
}

// This works as expected with no perf gain
// pub fn matmul_4x4_faer_complex(mut c: MatMut<c64>, a: MatRef<c64>, b: MatRef<c64>)
// {
//    let (r0, r1) = (0, 4);
//     for i in r0..r1 {
//         for j in r0..r1 {
//             // rely on this being additive id.
//             let mut s = c64::new(0.0, 0.0);
//             for k in r0..r1 {
//                 s += a[(i, k)] * b[(k, j)];
//             }
//             c[(i, j)] = s;
//         }
//     }
// }
