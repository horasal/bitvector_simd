use std::{
    fmt::Display,
    ops::{BitAnd, BitOr, BitXor, Index, Not},
};

//#[cfg(target_pointer_width = "64")]
use packed_simd::u64x8;

// BitContainer is the basic building block for internal storage
// BitVector will always aligned by BitContainer::bits
//#[cfg(target_pointer_width = "64")]
type BitContainer = u64x8;

//#[cfg(target_pointer_width = "32")]
//use packed_simd::u32x16;
//#[cfg(target_pointer_width = "32")]
//type BitContainer = u32x16;

#[derive(Debug, Clone)]
pub struct BitVector {
    // internal representation of bitvector
    storage: Vec<BitContainer>,
    // actual number of bits exists in storage
    nbits: usize,
}

/// Proc macro can not export BitVector
/// macro_rules! can not cancot ident
/// so we use name, name_2 for function names
macro_rules! impl_operation {
    ($name:ident, $name_2:ident, $op:tt) => {
        pub fn $name(self, other: Self) -> Self {
        assert_eq!(self.nbits, other.nbits);
        let storage = self
            .storage
            .into_iter()
            .zip(other.storage.into_iter())
            .map(|(a, b)| a $op b)
            .collect();
        Self {
            storage: storage,
            nbits: self.nbits,
        }
        }
        pub fn $name_2(&self, other: &Self) -> Self {
        assert_eq!(self.nbits, other.nbits);
        let storage = self
            .storage
            .iter()
            .cloned()
            .zip(other.storage.iter().cloned())
            .map(|(a, b)| a $op b)
            .collect();
        Self {
            storage: storage,
            nbits: self.nbits,
        }
        }
    };
}

// convert total bit to length
// input: Number of bits
// output:
//
// 1. the number of Vector used
// 2. after filling 1, the remaining bytes should be filled
// 3. after filling 2, the remaining bits should be filled
//
// notice that this result represents the length of vector
// so if 3. is 0, it means no extra bits after filling bytes
// return (length of storage, u64 of last container, bit of last elem)
// any bits > length of last elem should be set to 0
#[inline]
fn bit_to_len(nbits: usize) -> (usize, usize, usize) {
    (nbits / 512, (nbits % 512) / 64, nbits % 64)
}

#[test]
fn test_bit_to_len() {
    // contain nothing
    assert_eq!(bit_to_len(0), (0, 0, 0));
    assert_eq!(bit_to_len(1), (0, 0, 1));
    // 64bit only stores in a u64
    assert_eq!(bit_to_len(64), (0, 1, 0));
    // extra bit stores in extra u64
    assert_eq!(bit_to_len(65), (0, 1, 1));
    // 512bit only stores in a vector
    assert_eq!(bit_to_len(512), (1, 0, 0));
    assert_eq!(bit_to_len(513), (1, 0, 1));
    assert_eq!(bit_to_len(512 + 65), (1, 1, 1));
}

#[inline]
fn set_bit(flag: bool, bytes: u64, offset: u32) -> u64 {
    if flag {
        // set bit
        bytes | (1u64 << offset)
    } else {
        // clear bit
        bytes & !(1u64 << offset)
    }
}

impl BitVector {
    pub fn zeros(nbits: usize) -> Self {
        let (len, bytes, bits) = bit_to_len(nbits);
        let len = if bytes > 0 || bits > 0 { len + 1 } else { len };
        let storage = (0..len).map(|_| BitContainer::splat(0)).collect();
        Self {
            storage: storage,
            nbits: nbits,
        }
    }

    pub fn ones(nbits: usize) -> Self {
        let (len, bytes, bits) = bit_to_len(nbits);
        let mut storage = (0..len)
            .map(|_| BitContainer::splat(u64::MAX))
            .collect::<Vec<_>>();
        if bytes > 0 || bits > 0 {
            let slice = (0..bytes as u64)
                .map(|_| u64::MAX)
                .chain([(u64::MAX >> (u64::BITS - bits as u32) << (u64::BITS - bits as u32))])
                .chain((0..(512 / u64::BITS) - bytes as u32 - 1).map(|_| 0))
                .collect::<Vec<_>>();
            assert_eq!(slice.len(), 8);
            storage.push(BitContainer::from_slice_unaligned(&slice));
        }
        Self {
            storage: storage,
            nbits: nbits,
        }
    }

    pub fn shrink_to(&mut self, length: usize) {
        if length < self.nbits {
            let (i, bytes, bits) = bit_to_len(length);
            let mut storage = self.storage.drain(0..i).collect::<Vec<_>>();
            if bytes > 0 || bits > 0 {
                if let Some(s) = self.storage.drain(..).next() {
                    let mut s = s.replace(
                        bytes,
                        s.extract(bytes) >> (u64::BITS - bits as u32) << (u64::BITS - bits as u32),
                    );
                    for byte_index in (bytes + 1)..8 {
                        s = s.replace(byte_index, 0);
                    }
                    storage.push(s);
                } else {
                    panic!("incorrect internal representation of self")
                }
            }
            self.storage = storage;
            self.nbits = length;
        }
    }

    pub fn set(&mut self, index: usize, flag: bool) {
        let (i, bytes, bits) = bit_to_len(index);
        if self.nbits <= index {
            let i = if bytes > 0 || bits > 0 { i + 1 } else { i };
            self.storage
                .extend((0..i - self.storage.len()).map(|_| BitContainer::splat(0)));
            self.nbits = index + 1;
        }
        let byte = self.storage[i].extract(bytes);
        let byte = set_bit(flag, byte, u64::BITS - bits as u32 - 1);
        self.storage[i] = self.storage[i].replace(bytes, byte);
    }

    pub fn get(&self, index: usize) -> Option<bool> {
        if self.nbits <= index {
            None
        } else {
            let (index, bytes, bits) = bit_to_len(index);
            Some((self.storage[index].extract(bytes) & (1u64 << (u64::BITS - bits as u32 - 1))) > 0)
        }
    }

    pub fn get_unchecked(&self, index: usize) -> bool {
        if self.nbits <= index {
            panic!("index out of bounds {} > {}", index, self.nbits);
        } else {
            let (index, bytes, bits) = bit_to_len(index);
            (self.storage[index].extract(bytes) & (1u64 << (u64::BITS - bits as u32 - 1))) > 0
        }
    }

    impl_operation!(and, and_cloned, &);
    impl_operation!(or, or_cloned, |);
    impl_operation!(xor, xor_cloned, ^);

    pub fn difference(self, other: Self) -> Self {
        self.and(other.not())
    }

    pub fn difference_clone(&self, other: &Self) -> Self {
        // FIXME: This implementation has one extra clone
        self.and_cloned(&other.clone().not())
    }

    // not should make sure bits > nbits is 0
    pub fn not(self) -> Self {
        let (i, bytes, bits) = bit_to_len(self.nbits);
        let mut storage = self.storage.into_iter().map(|x| !x).collect::<Vec<_>>();
        if bytes > 0 || bits > 0 {
            assert_eq!(storage.len(), i + 1);
            if let Some(s) = storage.get_mut(i) {
                *s = s.replace(
                    bytes,
                    s.extract(bytes) >> (u64::BITS - bits as u32) << (u64::BITS - bits as u32),
                );
                for index in (bytes + 1)..8 {
                    *s = s.replace(index, 0);
                }
            } else {
                panic!("incorrect internal representation of self")
            }
        }

        Self {
            storage: storage,
            nbits: self.nbits,
        }
    }

    pub fn count(&self) -> usize {
        self.storage
            .iter()
            .map(|x| x.count_ones().wrapping_sum())
            .sum::<u64>() as usize
    }

    pub fn any(&self) -> bool {
        self.storage
            .iter()
            .any(|x| x.count_ones().max_element() > 0)
    }

    pub fn all(&self) -> bool {
        self.count() == self.nbits
    }

    pub fn none(&self) -> bool {
        !self.any()
    }

    pub fn eq_left(&self, other: &Self, bits: usize) -> bool {
        assert!(self.nbits >= bits && other.nbits >= bits);
        let (i, bytes, bits) = bit_to_len(bits);
        let r = self
            .storage
            .iter()
            .zip(other.storage.iter())
            .take(i)
            .all(|(a, b)| a == b);
        println!("{}", r);
        if bytes > 0 || bits > 0 {
            if let (Some(a), Some(b)) = (self.storage.get(i), other.storage.get(i)) {
                r && (0..bytes).all(|index| a.extract(index) == b.extract(index))
                    && ((a.extract(bytes) >> (u64::BITS - bits as u32))
                        == (b.extract(bytes) >> (u64::BITS - bits as u32)))
            } else {
                panic!("incorrect internal representation between self and other")
            }
        } else {
            r
        }
    }
}

impl Index<usize> for BitVector {
    type Output = bool;
    fn index(&self, index: usize) -> &Self::Output {
        if self.get_unchecked(index) {
            &true
        } else {
            &false
        }
    }
}

impl Display for BitVector {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (i, bytes, bits) = bit_to_len(self.nbits);
        for index in 0..i {
            let s = self.storage[index];
            for u in 0..8 {
                write!(f, "{:064b} ", s.extract(u))?;
            }
        }
        if bytes > 0 || bits > 0 {
            let s = self.storage[i];
            for u in 0..bytes {
                write!(f, "{:064b} ", s.extract(u))?;
            }
            write!(f, "{:064b}", s.extract(bytes))
        } else {
            Ok(())
        }
    }
}

impl PartialEq for BitVector {
    // eq should always ignore the bits > nbits
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.nbits, other.nbits);
        self.storage
            .iter()
            .zip(other.storage.iter())
            .all(|(a, b)| a == b)
    }
}

impl BitAnd for BitVector {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        self.and(rhs)
    }
}

impl BitOr for BitVector {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.or(rhs)
    }
}

impl BitXor for BitVector {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(rhs)
    }
}

impl Not for BitVector {
    type Output = Self;
    fn not(self) -> Self::Output {
        self.not()
    }
}

#[test]
fn test_bit_vec_eqleft() {
    let mut bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    assert!(bitvec.eq_left(&bitvec2, 1000));
    bitvec.set(900, false);
    println!("{}", bitvec);
    println!("{}", bitvec2);
    assert!(bitvec.eq_left(&bitvec2, 900));
    assert!(bitvec.eq_left(&bitvec2, 800));
    assert!(bitvec.eq_left(&bitvec2, 900));
    assert!(!bitvec.eq_left(&bitvec2, 901));
    assert!(!bitvec.eq_left(&bitvec2, 1000));
}

#[test]
fn test_bit_vec_count() {
    let mut bitvec = BitVector::ones(1000);
    assert_eq!(bitvec.count(), 1000);
    bitvec.set(1500, true);
    assert_eq!(bitvec.count(), 1001);
    bitvec.shrink_to(500);
    assert_eq!(bitvec.count(), 500);
}

#[test]
fn test_bit_vec_all_any() {
    let mut bitvec = BitVector::ones(1000);
    assert!(bitvec.all());
    assert!(bitvec.any());
    assert!(!bitvec.none());
    bitvec.set(10, false);
    assert!(!bitvec.all());
    assert!(bitvec.any());
    assert!(!bitvec.none());
    bitvec.set(1500, true);
    assert!(!bitvec.all());
    assert!(bitvec.any());
    assert!(!bitvec.none());
    let mut bitvec = BitVector::zeros(1000);
    assert!(!bitvec.all());
    assert!(!bitvec.any());
    assert!(bitvec.none());
    bitvec.set(1500, true);
    assert!(!bitvec.all());
    assert!(bitvec.any());
    assert!(!bitvec.none());
}

#[test]
fn test_bitvec_and_xor() {
    let bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    let bitvec3 = BitVector::zeros(1000);
    assert_eq!(bitvec.xor_cloned(&bitvec2), BitVector::zeros(1000));
    assert_eq!(bitvec.xor_cloned(&bitvec3), BitVector::ones(1000));
    assert_eq!(bitvec ^ bitvec2, BitVector::zeros(1000));
    let mut bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    bitvec.set(400, false);
    let bitvec3 = bitvec ^ bitvec2;
    assert!(bitvec3[400]);
    assert_eq!(bitvec3.count(), 1);
}

#[test]
fn test_bitvec_and_or() {
    let bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    let bitvec3 = BitVector::zeros(1000);
    assert_eq!(bitvec.or_cloned(&bitvec2), BitVector::ones(1000));
    assert_eq!(bitvec.or_cloned(&bitvec3), BitVector::ones(1000));
    assert_eq!(bitvec | bitvec2, BitVector::ones(1000));
    let mut bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    bitvec.set(400, false);
    let bitvec3 = bitvec | bitvec2;
    assert!(bitvec3.get_unchecked(400));
    assert_eq!(bitvec3.count(), 1000);
}

#[test]
fn test_bitvec_and_and() {
    let bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    let bitvec3 = BitVector::zeros(1000);
    assert_eq!(bitvec.and_cloned(&bitvec2), BitVector::ones(1000));
    assert_eq!(bitvec.and_cloned(&bitvec3), BitVector::zeros(1000));
    assert_eq!(bitvec & bitvec2, BitVector::ones(1000));
    let mut bitvec = BitVector::ones(1000);
    let bitvec2 = BitVector::ones(1000);
    bitvec.set(400, false);
    let bitvec3 = bitvec & bitvec2;
    assert!(!bitvec3.get_unchecked(400));
    assert_eq!(bitvec3.count(), 1000 - 1);
}

#[test]
fn test_bitvec_shrink() {
    let mut bitvec = BitVector::ones(1000);
    bitvec.shrink_to(900);
    assert_eq!(bitvec, BitVector::ones(900));
    bitvec.set(2000, true);
    assert!(bitvec.get_unchecked(2000));
    bitvec.shrink_to(1000);
    let mut bitvec2 = BitVector::ones(900);
    bitvec2.set(999, false);
    assert_eq!(bitvec, bitvec2);
}

#[test]
fn test_bitvec_not() {
    let bitvec = BitVector::ones(1000);
    assert_eq!(bitvec, BitVector::ones(1000));
    assert_eq!(bitvec.not(), BitVector::zeros(1000));
}

#[test]
fn test_bitvec_eq() {
    let mut bitvec = BitVector::ones(1000);
    assert_eq!(bitvec, BitVector::ones(1000));
    assert_ne!(bitvec, BitVector::zeros(1000));
    bitvec.set(50, false);
    assert_ne!(bitvec, BitVector::ones(1000));
    bitvec.set(50, true);
    assert_eq!(bitvec, BitVector::ones(1000));
}

#[test]
fn test_bitvec_creation() {
    let mut bitvec = BitVector::zeros(1000);
    for i in 0..1500 {
        if i < 1000 {
            assert_eq!(bitvec.get(i), Some(false));
        } else {
            assert_eq!(bitvec.get(i), None);
        }
    }
    bitvec.set(900, true);
    for i in 0..1500 {
        if i < 1000 {
            if i == 900 {
                assert_eq!(bitvec.get(i), Some(true));
            } else {
                assert_eq!(bitvec.get(i), Some(false));
            }
        } else {
            assert_eq!(bitvec.get(i), None);
        }
    }
    bitvec.set(1300, true);
    for i in 0..1500 {
        if i <= 1300 {
            if i == 900 || i == 1300 {
                assert_eq!(bitvec.get(i), Some(true));
            } else {
                assert_eq!(bitvec.get(i), Some(false));
            }
        } else {
            assert_eq!(bitvec.get(i), None);
        }
    }
}
