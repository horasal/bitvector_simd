## bitvector\_simd

[![creates.io(https://img.shields.io/crates/v/bitvector-simd.svg)](https://crates.io/crates/bitvector_simd)
[![docs.rs](https://docs.rs/bitvector_simd/badge.svg?version=0.1.1)](https://docs.rs/bitvector_simd/0.1.1/bitvector_simd/)


A bitvector implemented with [Packed SIMD 2](https://rust-lang.github.io/packed_simd/packed_simd_2/).

## How to use


```rust

let mut bitvec = BitVector::ones(1000); // create a bitvector contains 0 ..= 999
bitvec.set(900, false); // delete 900 from bitvector
bitvec.set(1200, true); // add 1200 to bitvector (and expand bitvector to length 1201)
let bitvec2 = BitVector::ones(1000);

let new_bitvec = bitvec.and_cloned(&bitvec2); // and operation, without consume
let new_bitvec2 = bitvec & bitvec2; // and operation, consume both bitvector

// Operation Supported:
// and, or, xor, not, eq, eq_left

assert_eq!(new_bitvec, new_bitvec2);
```

## Performance 

Compared with

* [bit\_vec 0.6.3](https://docs.rs/bit-vec/0.6.3/bit_vec/index.html)
* [bitvec 0.22.3](https://docs.rs/bitvec/0.22.3/bitvec/index.html)


```
$ cargo bench       

Benchmarking BitVector SIMD(this crate): Collecting 100 samples in estimated 5.0013 s (9.0M iteratio                                                                                                    BitVector SIMD(this crate)                        
                        time:   [529.56 ns 531.50 ns 533.41 ns]
Found 8 outliers among 100 measurements (8.00%)
  1 (1.00%) low severe
  1 (1.00%) low mild
  5 (5.00%) high mild
  1 (1.00%) high severe

bit-vec 0.6             time:   [3.4030 us 3.4135 us 3.4250 us]                         
Found 7 outliers among 100 measurements (7.00%)
  1 (1.00%) low mild
  2 (2.00%) high mild
  4 (4.00%) high severe

bitvec 0.22             time:   [609.61 us 611.75 us 614.43 us]                        
Found 5 outliers among 100 measurements (5.00%)
  2 (2.00%) low mild
  1 (1.00%) high mild
  2 (2.00%) high severe

Benchmarking BitVector SIMD(this crate) with creation: Collecting 100 samples in estimated 5.0033 s                                                                                                     BitVector SIMD(this crate) with creation                        
                        time:   [1.0174 us 1.0205 us 1.0237 us]
Found 5 outliers among 100 measurements (5.00%)
  1 (1.00%) low mild
  3 (3.00%) high mild
  1 (1.00%) high severe

Benchmarking bit-vec 0.6 with creation: Collecting 100 samples in estimated 5.0008 s (6.3M iteration                                                                                                    bit-vec 0.6 with creation                        
                        time:   [796.80 ns 799.33 ns 802.00 ns]
Found 4 outliers among 100 measurements (4.00%)
  2 (2.00%) low mild
  2 (2.00%) high mild

Benchmarking bitvec 0.22 with creation: Collecting 100 samples in estimated 5.4980 s (20k iterations                                                                                                    bitvec 0.22 with creation                        
                        time:   [270.70 us 271.38 us 272.15 us]
Found 5 outliers among 100 measurements (5.00%)
  1 (1.00%) low mild
  2 (2.00%) high mild
  2 (2.00%) high severe
```
