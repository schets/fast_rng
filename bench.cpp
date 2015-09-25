#include <iostream>
#include <time.h>

#include "xorshift.hpp"
constexpr static size_t num_loop = 1e8;
constexpr static int num_work = num_loop;
using namespace std;

template<class seeder>
void seed_rn(seeder &rn) {
	rn.seed(0xdeadbeef, 0xcafebabe);
}

//this forces the compiler to actually
//do the full rng execution
//when inside of the inner loop,
//if this call is inlined,
//then the compiler can optimize away a bunch of stuff in each case
//for the first one, some of the reads/writes to memory it generally incurs
//for the second two, probably can unroll the if statement,
//and the write to the index

//small loop to remove some cost of function call
template<class rng>
void __attribute__ ((noinline))  call_rn (rng &rn) {
    rn.rand();
}

template<class rng>
void __attribute__ ((noinline)) work_rn(rng &rn) {
    rn.do_heavy_work();
}

template<class rng>
void grab_many(rng& rn) {
	for (size_t i = 0; i < num_loop; i++) {
		call_rn(rn);
	}
}

template<class rng>
void heavy_work(rng& rn) {
	for (size_t i = 0; i < num_work; i++) {
		work_rn(rn);
	}
}

template<class FN>
uint64_t time_call(const FN& f, double &total) {
	clock_t t = clock();
	f();
	t = clock() - t;
	total = t * 1.0/CLOCKS_PER_SEC;
	return 0;
}


int main() {
	cout << "timing rngs" << std::endl;

	double time;
    double lat_time;
	uint64_t junk_val = 0;
	junk_val += time_call([]() {
		xorshift<uint64_t> rn;
		seed_rn(rn);
		return grab_many(rn);
	}, time);

	cout << "The plain rng took " << time << " seconds, ";
    cout << (num_loop * 1.0/time) << " rngs per second " << endl << endl;

    junk_val += time_call([]() {
        xorshift_bulk<uint16_t, 8> rn;
        seed_rn(rn);
        return grab_many(rn);
    }, time);

    cout << "The bulk rng took " << time << " seconds, ";
    cout << (num_work * 1.0/time) << " rngs per second " << endl;

     junk_val += time_call([]() {
        xorshift_bulk<uint16_t, 8> rn;
        seed_rn(rn);
        heavy_work (rn);
    }, time);

    cout << "The bulk rng took " << time << " seconds for refilling, ";
    cout << (num_work * 1.0/time) << " refills per second ";
    cout << "and " << (1e9 * time / num_work) << "ns per refill" << endl << endl;

    junk_val += time_call([]() {
        alignas(64) xorshift_sse2<uint16_t, 4> rn;
        seed_rn(rn);
        return grab_many(rn);
    }, time);

    cout << "The vector rng took " << time << " seconds, ";
    cout << (num_loop * 1.0/time) << " rngs per second " << endl;


    junk_val += time_call([]() {
       alignas(64) xorshift_sse2<uint64_t, 4> rn;
        seed_rn(rn);
        heavy_work (rn);
    }, time);

    cout << "The vector rng took " << time << " seconds for refilling, ";
    cout << (num_work * 1.0/time) << " refills per second ";
    cout << "and " << (1e9 * time / num_work) << "ns per refill" << endl << endl;

    cout << "Caveat - all these numbers are in a hot loop" << endl;
}