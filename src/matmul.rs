pub trait Index2<T> {
    fn index2(&self, a: usize, b: usize) -> T;
}

pub trait Assign<T> {
    fn assign(&mut self, a: usize, b: usize, val: T);
}

/// Return the additive identity element
pub trait MyZero {
    type Output;
    fn my_zero() -> Self::Output;
}

impl MyZero for f64 {
    type Output = f64;
    #[inline(always)]
    fn my_zero() -> f64 {
        0.0
    }
}

pub trait MyNrows {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

/// Row major just means that the row index varies fastest.
pub fn matmul_2x2_row_major<T, MT1, MT2, MTM>(mut c: MTM, a: MT1, b: MT2)
where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T> + Copy,
      MT1: Index2<T> + MyNrows, MT2: Index2<T> + MyNrows, MTM: MyNrows + Assign<T> {
    let (r0, r1) = (0, 2);
    for j in r0..r1 {
        for i in r0..r1 {
            let mut s = T::my_zero();
            for k in r0..r1 {
                s += a.index2(i, k) * b.index2(k, j);
            }
            c.assign(i, j, s);
        }
    }
}

// We expect that the compiler will unroll the explicit loops so that this is not necessary.
// Benchmarking appears to verify this.
pub fn matmul_2x2_row_major_unrolled<T, MT1, MT2, MTM>(mut c: MTM, a: MT1, b: MT2)
where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T> + Copy + std::ops::Add<Output = T>,
      MT1: Index2<T> + MyNrows, MT2: Index2<T> + MyNrows, MTM: MyNrows + Assign<T> {
    c.assign(0, 0, a.index2(0, 0) * b.index2(0, 0) + a.index2(0, 1) * b.index2(1, 0));
    c.assign(0, 1, a.index2(0, 0) * b.index2(0, 1) + a.index2(0, 1) * b.index2(1, 1));
    c.assign(1, 0, a.index2(1, 0) * b.index2(0, 0) + a.index2(1, 1) * b.index2(1, 0));
    c.assign(1, 1, a.index2(1, 0) * b.index2(0, 1) + a.index2(1, 1) * b.index2(1, 1));
}

// Matrix multiplication intended to be generic wrt matrix libraries.
// That is, implement traits for each linear algebra library and then call this
// single method.
// The main features are:
// 1. Naive explicit loops
// 2. Hard-coded limits on loops, so the compiler has an opportunity to perform optimizations
//    such as loop unrolling.
/// Row major just means that the row index varies fastest.
pub fn matmul_4x4_row_major<T, MT1, MT2, MTM>(mut c: MTM, a: MT1, b: MT2)
where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T> + Copy,
      MT1: Index2<T> + MyNrows, MT2: Index2<T> + MyNrows, MTM: MyNrows + Assign<T> {
    let (r0, r1) = (0, 4);
    for j in r0..r1 {
        for i in r0..r1 {
            let mut s = T::my_zero();
            for k in r0..r1 {
                s += a.index2(i, k) * b.index2(k, j);
            }
            c.assign(i, j, s);
        }
    }
}

/// Column major just means that the column index varies fastest.
pub fn matmul_4x4_col_major<T, MT1, MT2, MTM>(mut c: MTM, a: MT1, b: MT2)
where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T> + Copy,
      MT1: Index2<T> + MyNrows, MT2: Index2<T> + MyNrows, MTM: MyNrows + Assign<T> {
    let (r0, r1) = (0, 4);
    for i in r0..r1 {
        for j in r0..r1 {
            let mut s = T::my_zero();
            for k in r0..r1 {
                s += a.index2(i, k) * b.index2(k, j);
            }
            c.assign(i, j, s);
        }
    }
}

// We really want to return a `Result` here. But this is ok for a first test.
pub fn matmul_4x4_checked_row_major<T, MT, MTM>(c: MTM, a: MT, b: MT) -> bool
where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T> + Copy,
      MT: Index2<T> + MyNrows, MTM: MyNrows + Assign<T> {
    let nside = a.ncols();
    if nside != 4 || nside != a.ncols() {
        return false
    }
    matmul_4x4_row_major(c, a, b);
    true
}

// The difference from matmul_4x4 is that the size of the matrix is retrieved
// at run time rather than compile time.
pub fn matmul_nxn_row_major<T, MT, MTM>(mut c: MTM, a: MT, b: MT)
where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T> + Copy,
      MT: Index2<T> + MyNrows, MTM: MyNrows + Assign<T> {
    let nside = a.ncols();
    for j in 0..nside {
        for i in 0..nside {
            let mut s = T::my_zero();
            for k in 0..nside {
                s += a.index2(i, k) * b.index2(k, j);
            }
            c.assign(i, j, s);
        }
    }
}

// The difference from matmul_4x4 is that the size of the matrix is retrieved
// at run time rather than compile time.
pub fn matmul_nxn_col_major<T, MT, MTM>(mut c: MTM, a: MT, b: MT)
where T: std::ops::Mul<Output = T> + std::ops::AddAssign + MyZero<Output = T> + Copy,
      MT: Index2<T> + MyNrows, MTM: MyNrows + Assign<T> {
    let nside = a.ncols();
    for i in 0..nside {
        for j in 0..nside {
            let mut s = T::my_zero();
            for k in 0..nside {
                s += a.index2(i, k) * b.index2(k, j);
            }
            c.assign(i, j, s);
        }
    }
}
