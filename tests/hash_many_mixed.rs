//! `hash_many` on inputs that no equal-length run can fuse, against the
//! independent RustCrypto oracle. These shapes exercise the lane-refill
//! scheduler: lanes retire and are refilled at different times, padding blocks
//! of one message share a round with full blocks of another, and the batch can
//! be larger than one lane window.

use md5::Digest as _;
use md5_simd::Md5Engine;

mod common;
use common::run_with_large_stack;

fn oracle(data: &[u8]) -> [u8; 16] {
    md5::Md5::digest(data).into()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next() as u8).collect()
    }
}

fn check(storage: &[Vec<u8>], what: &str) {
    let inputs: Vec<&[u8]> = storage.iter().map(Vec::as_slice).collect();
    let mut outputs = vec![[0u8; 16]; inputs.len()];
    Md5Engine::new().hash_many(&inputs, &mut outputs);
    for (i, (input, output)) in inputs.iter().zip(&outputs).enumerate() {
        assert_eq!(
            *output,
            oracle(input),
            "{what}: message {i} len {}",
            input.len()
        );
    }
}

#[test]
fn all_distinct_lengths_around_block_boundaries() {
    run_with_large_stack(|| {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        // 0..=130 covers empty, one/two padding blocks, and the 55/56/63/64 edges,
        // each length once so no two adjacent messages can fuse.
        let storage: Vec<Vec<u8>> = (0..=130).map(|len| rng.bytes(len)).collect();
        check(&storage, "0..=130");
        let mut reversed = storage.clone();
        reversed.reverse();
        check(&reversed, "130..=0");
    });
}

#[test]
fn random_lengths_from_tiny_to_megabytes() {
    run_with_large_stack(|| {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for round in 0..12 {
            let count = 1 + rng.below(200);
            let storage: Vec<Vec<u8>> = (0..count)
                .map(|_| {
                    let len = match rng.below(4) {
                        0 => rng.below(70),
                        1 => rng.below(5_000),
                        2 => 65_536 + rng.below(200),
                        _ => rng.below(1_500_000),
                    };
                    rng.bytes(len)
                })
                .collect();
            check(&storage, &format!("random round {round} count {count}"));
        }
    });
}

#[test]
fn one_long_message_among_many_short_ones() {
    run_with_large_stack(|| {
        let mut rng = Rng(0x0bad_cafe_1234_5678);
        // The long message keeps its lane while short ones retire and refill
        // around it, so every round advances it by a block or two.
        let mut storage: Vec<Vec<u8>> = (0..300).map(|i| rng.bytes(64 + i)).collect();
        storage.insert(7, rng.bytes(2_000_000));
        check(&storage, "long among short");
    });
}

#[test]
fn leftovers_below_the_simd_threshold() {
    run_with_large_stack(|| {
        let mut rng = Rng(0x1357_9bdf_2468_ace0);
        for count in 1..=3 {
            let storage: Vec<Vec<u8>> = (0..count).map(|i| rng.bytes(1000 + 77 * i)).collect();
            check(&storage, &format!("{count} unequal messages"));
        }
        // An equal run of 8 (fused path) followed by 3 leftovers of other lengths.
        let mut storage: Vec<Vec<u8>> = (0..8).map(|_| rng.bytes(4096)).collect();
        storage.extend((0..3).map(|i| rng.bytes(500 + i)));
        check(&storage, "fused run then leftovers");
    });
}

#[test]
fn outputs_beyond_inputs_are_untouched() {
    let storage: Vec<Vec<u8>> = (0..9).map(|i| vec![i as u8; 100 + 3 * i]).collect();
    let inputs: Vec<&[u8]> = storage.iter().map(Vec::as_slice).collect();
    let mut outputs = vec![[0xAAu8; 16]; 12];
    Md5Engine::new().hash_many(&inputs, &mut outputs);
    assert!(outputs[9..].iter().all(|out| *out == [0xAA; 16]));
    assert!(
        inputs
            .iter()
            .zip(&outputs)
            .all(|(input, out)| *out == oracle(input))
    );
}
