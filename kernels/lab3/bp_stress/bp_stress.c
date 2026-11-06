#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static uint64_t next_random(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

int main(int argc, char **argv) {
    long size = 1L << 20;
    int iters = 300;
    unsigned long long seed = 42;
    int random_branch = 1;

    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--size"))
            size = atol(argv[++i]);
        else if (!strcmp(argv[i], "--iterations"))
            iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed"))
            seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--random"))
            random_branch = atoi(argv[++i]);
    }

    if (size < 1 || (uint64_t)size > SIZE_MAX / sizeof(size_t) ||
        iters < 1 || (random_branch != 0 && random_branch != 1)) {
        puts("status: FAILED\nerror: invalid size, iterations, or random flag");
        return 1;
    }

    size_t n = (size_t)size;
    size_t *order = malloc(n * sizeof(*order));
    size_t *next = malloc(n * sizeof(*next));
    uint8_t *branch_bits = malloc(n * sizeof(*branch_bits));
    if (!order || !next || !branch_bits) {
        free(order);
        free(next);
        free(branch_bits);
        puts("status: FAILED\nerror: allocation failed");
        return 1;
    }

    uint64_t rng = seed ? (uint64_t)seed : UINT64_C(1);

    // Shuffle the indices, then link them into one cycle.
    for (size_t i = 0; i < n; i++)
        order[i] = i;

    for (size_t i = n - 1; i > 0; i--) {
        size_t j = next_random(&rng) % (i + 1);
        size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }

    for (size_t i = 1; i < n; i++)
        next[order[i - 1]] = order[i];
    next[order[n - 1]] = order[0];

    size_t idx = order[0];
    free(order);

    // Each index gets one fixed branch decision for the whole run.
    for (size_t i = 0; i < n; i++) {
        uint8_t bit = (uint8_t)(next_random(&rng) >> 63);
        branch_bits[i] = random_branch ? bit : 1;
    }

    uint64_t acc = 1;
    double t0 = now_us();

    for (int it = 0; it < iters; it++) {
        for (size_t i = 0; i < n; i++) {
            idx = next[idx];
            if (branch_bits[idx]) {
                __asm__ volatile("add %0, %0, %1" : "+r"(acc) : "r"(idx));
            } else {
                __asm__ volatile("eor %0, %0, %1" : "+r"(acc) : "r"(idx));
            }
        }
    }

    double us = now_us() - t0;
    free(next);
    free(branch_bits);

    printf("status: OK\n");
    printf("size: %ld\niterations: %d\n", size, iters);
    printf("mean_us: %.1f\n", us / iters);
    printf("wall_us: %.1f\n", us);
    printf("flop: 0\n");
    printf("fingerprint: %llu\n", (unsigned long long)acc);
    return 0;
}