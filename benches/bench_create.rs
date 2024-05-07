use criterion::{black_box, criterion_group, criterion_main, Criterion};

use small_matrix_rust::{
    gates::czgate, gates::czgate3,gates::czgate4,
//     gates::czgate2
};

fn czgate_bench(crit: &mut Criterion) {
    crit.bench_function("czgate",
                        |bench| bench.iter(|| czgate()));
}

// fn czgate2_bench(crit: &mut Criterion) {
//     crit.bench_function("czgate2",
//                         |bench| bench.iter(|| czgate2()));
// }

fn czgate3_bench(crit: &mut Criterion) {
    crit.bench_function("czgate3",
                        |bench| bench.iter(|| czgate3()));
}

fn czgate4_bench(crit: &mut Criterion) {
    crit.bench_function("czgate4",
                        |bench| bench.iter(|| czgate4()));
}


criterion_group!(benches,
                 // czgate3_bench,
                 // czgate2_bench,
                 czgate4_bench,
                 czgate_bench,
                 );

criterion_main!(benches);
