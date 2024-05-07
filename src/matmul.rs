use num_complex::Complex64;
use ndarray::{ArrayView2, ArrayViewMut2};
use faer::prelude::*;
use faer_entity::IdentityGroup;
// use faer::{Mat, Parallelism, Entity};
use faer::{Entity};
use std::ops::IndexMut;

pub trait QIndex<T> {
    fn qindex2(&self, a: usize, b: usize) -> T;
}

pub trait QIndexMut<T> {
    type Output;
    fn qindexmut2(&mut self, a: usize, b: usize) -> &mut Self::Output;
}

// Matrix multiplication intended to be generic wrt matrix libraries.
// That is, implment traits for each linalg library and then call this
// single method.
pub fn matmul_4x4<T, MT, MTM>(mut c: MTM, a: MT, b: MT)
where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T> + Copy,
      MT: QIndex<T> + MyNrows, MTM: QIndexMut<T, Output = T> + MyNrows {
    // let nside = 4;
    // let n = nside - 1;
    for i in 0..3 {
        for j in 0..3 {
            // rely on this being additive id.
            let mut s = T::my_zero();
            for k in 0..3 {
                s += a.qindex2(i, k) * b.qindex2(k, j);
            }
            *c.qindexmut2(i, j) = s;
        }
    }
}

pub fn matmul_4x4_checked<T, MT, MTM>(c: MTM, a: MT, b: MT) -> bool
where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T> + Copy,
      MT: QIndex<T> + MyNrows, MTM: QIndexMut<T, Output = T> + MyNrows {
    let nside = a.ncols();
    if nside != 4 || nside != a.ncols() {
        return false
    }
    matmul_4x4(c, a, b);
    true
}

pub fn matmul_nxn<T, MT, MTM>(mut c: MTM, a: MT, b: MT)
where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T> + Copy,
      MT: QIndex<T> + MyNrows, MTM: QIndexMut<T, Output = T> + MyNrows {
    let nside = a.ncols();
    let n = nside - 1;
    for i in 0..n {
        for j in 0..n {
            // rely on this being additive id.
            let mut s = T::my_zero();
            for k in 0..n {
                s += a.qindex2(i, k) * b.qindex2(k, j);
            }
            *c.qindexmut2(i, j) = s;
        }
    }
}

/// Return the additive identity element
pub trait MyZero {
    type Output;
    fn my_zero() -> Self::Output;
}

// "literal" means that the loop limits are literal integers. This should
// give the same results as `let nside = 4`.
pub fn matmul_4x4_literal<T, MT, MTM>(mut c: MTM, a: MT, b: MT)
where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T> + Copy,
      MT: QIndex<T>, MTM: QIndexMut<T, Output = T> {
    for i in 0..3 {
        for j in 0..3 {
            // rely on this being additive id.
            let mut s = T::my_zero();
            for k in 0..3 {
                s += a.qindex2(i, k) * b.qindex2(k, j);
            }
            *c.qindexmut2(i, j) = s;
        }
    }
}

impl MyZero for f64 {
    type Output = f64;
    #[inline(always)]
    fn my_zero() -> f64 {
        0.0
    }
}

impl MyZero for Complex64 {
    type Output = Complex64;
    #[inline(always)]
    fn my_zero() -> Complex64 {
        Complex64::new(0.0, 0.0)
    }
}

impl MyZero for c64 {
    type Output = c64;
    #[inline(always)]
    fn my_zero() -> c64 {
        c64::new(0.0, 0.0)
    }
}

pub trait MyNrows {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

impl<A> MyNrows for ArrayView2<'_, A> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols()
    }
}

impl<A> MyNrows for ArrayViewMut2<'_, A> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols()
    }
}

impl<E: Entity > MyNrows for MatRef<'_, E> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols()
    }
}

impl<E: Entity > MyNrows for MatMut<'_, E> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols()
    }
}

//
// ndarray implementations
//

impl<T: Copy> QIndex<T> for ArrayView2<'_, T> {
    #[inline(always)]
    fn qindex2(&self, a: usize, b: usize) -> T {
        self[[a, b]]
    }
}

// For indexing in lvalues
impl<T: Copy> QIndexMut<T> for ArrayViewMut2<'_, T> {
    type Output = T;
    #[inline(always)]
    fn qindexmut2(&mut self, a: usize, b: usize) -> &mut T {
        self.index_mut([a,b])
    }
}

// Use indexing methods from ndarray library rather than those defined above.
pub fn matmul_4x4_array_view<T>(mut c: ArrayViewMut2<T>, a: ArrayView2<T>, b: ArrayView2<T>)
where T:  std::ops::Add<Output = T>  + std::ops::Mul<Output = T> + std::ops::AddAssign + Default + Copy
{
    for i in 0..3 {
        for j in 0..3 {
            // rely on this being additive id.
            let mut s = T::default();
            for k in 0..3 {
                s += a[[i, k]] * b[[k, j]];
            }
            c[[i, j]] = s;
        }
    }
}

//
// faer implementations
//

impl<T: Copy + Entity<Group = IdentityGroup, Unit=T>> QIndex<T> for MatRef<'_, T> {
    #[inline(always)]
    fn qindex2(&self, a: usize, b: usize) -> T {
        self[(a, b)]
    }
}

// For indexing in lvalues
impl <T: Copy + Entity<Group = IdentityGroup, Unit=T>> QIndexMut<T> for MatMut<'_, T> {
    type Output = T;
    #[inline(always)]
    fn qindexmut2(&mut self, a: usize, b: usize) -> &mut T {
        self.index_mut((a, b))
    }
}

// Use indexing methods from faer library rather than those defined above.
// In principle, the generic routine use zero-cost abstractions to obtain
// the same performance.
// This method with explicit-ish indexing may be used to verify that there are no
// inefficiencies in the generic version.
pub fn matmul_4x4_faer<T>(mut c: MatMut<T>, a: MatRef<T>, b: MatRef<T>)
where T:  std::ops::Add<Output = T>  + std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T>  + Copy
    + Entity<Group = IdentityGroup, Unit=T>
{
    for i in 0..3 {
        for j in 0..3 {
            // rely on this being additive id.
            let mut s = T::my_zero();
            for k in 0..3 {
                s += a[(i, k)] * b[(k, j)];
            }
            c[(i, j)] = s;
        }
    }
}
