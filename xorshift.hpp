#include <stdint.h>
#include <stddef.h>
#include <utility>

#include <emmintrin.h>

#define likely(x)    __builtin_expect(!!(x), 1)
#define unlikely(x)  __builtin_expect(!!(x), 0)

//need to test using a 64 bit compiler
template<class itype>
static inline itype do_rng(itype &s0, itype _s1) {
    auto s1 = _s1;
    s1 ^= s1 << 23;
    s1 = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26));
    _s1 = s1;
    return s1 + s0;
}

template<class int_type>
class xorshift {
    uint64_t state[2];

public:

    void seed(uint64_t a, uint64_t b) {
        state[0] = a;
        state[1] = b;
    }
    //does the work to represent a 'latency hit'
    //non in this case
    void do_heavy_work() {}
    int_type rand() {
        std::swap(state[0], state[1]);
        return do_rng(state[0], state[1]);
    }
};

//this RNG performs bulk initiation of the
//array using a more effecient loop
//but, does do some more memory stores
//by saving data in an array
//also is not vectorized
template<class int_type, size_t size64>
class xorshift_bulk {
protected:
    uint64_t data[size64];
    uint64_t state[2];
    uint16_t index;

    static constexpr size_t int_num = size64 * sizeof(uint64_t) / sizeof(int_type);

public:
    void seed(uint64_t a, uint64_t b) {
        state[0] = a;
        state[1] = b;
        do_heavy_work();
    }
    void do_heavy_work() {
        uint64_t s0 = state[0];
        uint64_t s1 = state[1];
        for(size_t i = 0; i < size64; i += 2) {
            data[i] = do_rng(s1, s0);
            data[i+1] = do_rng(s0, s1);
        }
        state[0] = s0;
        state[1] = s1;
        index = 0;
    }

    int_type rand() {
        if (unlikely(index == int_num)) {
            do_heavy_work();
        }
        return ((int_type *)data)[index++];
    }
};
#define psllqi(a, b) _mm_slli_epi64(a, b)
#define psrlqi(a, b) _mm_srli_epi64(a, b)

#define pxor(a, b) _mm_xor_si128(a, b)

#define padd(a, b) _mm_add_epi64(a, b)


//allowing gcc to do this by itself
//with normal operators on the vector type
//is an absolute disaster...

//this algorithm can trivially be extended to avx2 as well
//if the machine running even has it

//extern C for easy lookup in object file
extern "C" {

//avx2 to come once I get that working on windows...
static __m128i rng_sse2(__m128i &_s0, __m128i &_s1) {
    //s1 = s1 ^ (s1 << 23)
    __m128i s1 = _s1;
    __m128i temp1 = s1;
    temp1 = psllqi(s1, 23);
    s1 = pxor(s1, temp1);
    __m128i s0 = _s0;
    //s0 >> 26
    __m128i temp2 = psrlqi(s0, 26);
    //(s1 >> 17);
    temp1 = psrlqi(s1, 17);
    //temp2 ^ temp1 ... duh
    temp2 = pxor(temp2, temp1);
    //s1 ^ s0
    temp1 = pxor(s0, s1);
    //_s1 = ...
    s1 = pxor(temp2, temp1);
    return padd(s1, s0);
}
}

template<class int_type, size_t size128>
class xorshift_sse2 {
protected:
    __m128i _data[size128];
    __m128i state[2];
    uint16_t index;

    static constexpr size_t int_num = size128 * sizeof(__m128i) / sizeof(int_type);

public:
    void seed(uint64_t a, uint64_t b) {
        state[0] = _mm_set_epi64x(a, b);
        state[1] = _mm_set_epi64x(b, a);
        do_heavy_work();
    }

    void do_heavy_work() {
        __m128i s0 = state[0];
        __m128i s1 = state[1];
        __m128i * __restrict data =
                (__m128i *__restrict ) __builtin_assume_aligned(&_data[0], 16);


        for(size_t i = 0; i < size128; i += 2) {
            data[i] = rng_sse2(s1, s0);
            data[i+1] = rng_sse2(s0, s1);
        }
        state[0] = s0;
        state[1] = s1;
        index = 0;
    }

    int_type rand() {
        if (unlikely(index == int_num)) {
            do_heavy_work();
        }
        return ((int_type *)_data)[index++];
    }
};
