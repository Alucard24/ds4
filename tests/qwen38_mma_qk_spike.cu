/* Spike: quanto costa il QK^T dell'attenzione GA con i tensor core, contro il
 * corpo di release che gira a 0,246 ns per (chiave, riga) misurati stamattina.
 *
 * Come si costruisce e si legge (Linux, sm_120a):
 *
 *   nvcc -O3 --use_fast_math -arch=sm_120a -DMVARIANT=0 -o mma_0 \
 *       tests/qwen38_mma_qk_spike.cu && ./mma_0
 *   nvcc -O3 --use_fast_math -arch=sm_120a -DMVARIANT=1 -o mma_1 \
 *       tests/qwen38_mma_qk_spike.cu && ./mma_1
 *
 * Stampa i ns per (chiave, riga) del dot.  Misurati il 2026-09-12 su RTX 5070 Ti:
 * MVARIANT 0 = 0,1357 (forma di release), MVARIANT 1 = 0,0191 (HMMA m16n8k16),
 * cioe' 7,1x.  Il resto del corpo (softmax, accumulo V, carichi) vale 0,11 ns e
 * non cambia: da li' l'estrapolazione di 1,9x sull'attenzione.
 *
 * Il banco del corpo completo, che da il 0,246 di riferimento, e' lo stesso
 * programma stampato con PBODY=4 in /tmp/ga_micro.cu durante quella sessione; il
 * suo contenuto e' riprodotto dal commento sopra MVARIANT 0 qui sotto.
 *
 * Geometria identica alla kernel vera: 256 dimensioni per testa, tile di chiavi
 * in shared (16 byte alla volta), una testa per blocco.  Cambia solo la forma:
 *
 *   MVARIANT 0  il dot di release: 8 LDS.U16 + 8 conversioni + 8 FMA + la
 *               butterfly a 5 shuffle, un warp per riga (16 righe in 16 warp)
 *   MVARIANT 1  HMMA m16n8k16: un warp su 16 righe x 8 chiavi per tile, il dot
 *               sono 16 istruzioni per 128 punteggi invece di 41 per punteggio
 *
 * Il Q resta nei registri (frammento A) per tutto il giro delle chiavi: e' il
 * guadagno strutturale dell'MMA, e il motivo per cui vale la pena misurarlo.
 *
 * Non c'e' softmax qui: questo misura il dot, che e' la parte che l'MMA cambia.
 * Il softmax (exp) e l'accumulo V restano comuni alle due forme e vanno contati
 * a parte quando si decide.
 */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define DIM 256u
#define NKEYS 32u
#define NHEAD 24u
#define NBLK 1392u
#define REPEAT 8u

#ifndef MVARIANT
#define MVARIANT 1
#endif

__device__ static __forceinline__ float warp_sum_f32(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

/* --- variante 0: il dot di release, un warp per riga ---------------------- */
__global__ static void qk_scalar_kernel(const __half *kg, const __half *qg,
                                        float *out, uint32_t n_keys) {
    __shared__ __align__(16) __half k_tile[NKEYS * DIM];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    for (uint32_t i = threadIdx.x; i < NKEYS * DIM; i += blockDim.x)
        k_tile[i] = kg[i];
    __syncthreads();
    /* una riga per warp: il frammento che l'MMA userebbe per 16 righe, qui e'
     * spalmato su 16 warp (che e' esattamente la forma di release) */
    const uint32_t row = blockIdx.x * 16u + warp;
    const float *q = (const float *)qg + (uint64_t)row * DIM;
    float q0 = q[lane], q32 = q[lane + 32u], q64 = q[lane + 64u];
    float q96 = q[lane + 96u], q128 = q[lane + 128u];
    float q160 = q[lane + 160u], q192 = q[lane + 192u], q224 = q[lane + 224u];
    float sink = 0.0f;
    for (uint32_t rep = 0; rep < REPEAT; rep++) {
        for (uint32_t key = 0; key < n_keys; key++) {
            const uint64_t base = (uint64_t)key * DIM;
            float p0 = q0 * __half2float(k_tile[base + lane]);
            p0 += q128 * __half2float(k_tile[base + lane + 128u]);
            float p1 = q64 * __half2float(k_tile[base + lane + 64u]);
            p1 += q192 * __half2float(k_tile[base + lane + 192u]);
            p0 += p1;
            float p2 = q32 * __half2float(k_tile[base + lane + 32u]);
            p2 += q160 * __half2float(k_tile[base + lane + 160u]);
            float p3 = q96 * __half2float(k_tile[base + lane + 96u]);
            p3 += q224 * __half2float(k_tile[base + lane + 224u]);
            p2 += p3;
            p0 += p2;
            sink += warp_sum_f32(p0);
        }
    }
    if (sink == 12345.678f) out[0] = sink;
}

/* --- variante 1: HMMA m16n8k16 -------------------------------------------- */
/* Frammenti (PTX ISA, m16n8k16 f16 con accumulo f32):
 *   A (16x16, row-major): a0a1 = riga lane>>2, colonne (lane%4)*2 +{0,1};
 *                         a2a3 = riga+8; a4a5 = colonne+8; a6a7 = riga+8, col+8
 *   B (16x8, col-major):  b0b1 = righe (lane%4)*2 +{0,1}, colonna lane>>2;
 *                         b2b3 = righe+8
 *   C/D (16x8 f32):       c0c1 = riga lane>>2, colonne (lane%4)*2 +{0,1};
 *                         c2c3 = riga+8
 */
__device__ static __forceinline__ void hmma_m16n8k16(float *c, const __half *a,
                                                     const __half *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(*(const uint32_t *)&a[0]), "r"(*(const uint32_t *)&a[2]),
          "r"(*(const uint32_t *)&a[4]), "r"(*(const uint32_t *)&a[6]),
          "r"(*(const uint32_t *)&b[0]), "r"(*(const uint32_t *)&b[2]));
}

__global__ static void qk_mma_kernel(const __half *kg, const __half *qg,
                                     float *out, uint32_t n_keys) {
    __shared__ __align__(16) __half k_tile[NKEYS * DIM];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    for (uint32_t i = tid; i < NKEYS * DIM; i += blockDim.x)
        k_tile[i] = kg[i];
    __syncthreads();
    const uint32_t qrow = (lane >> 2);
    const uint32_t qcol = (lane & 3u) * 2u;
    /* Q: 16 righe x 256 dimensioni, frammenti A tenuti in registri per tutto il
     * giro (8 half per thread per k-step, 16 k-step) */
    __half qa[16][8];
    const uint32_t row0 = blockIdx.x * 16u;
    const float *qbase = (const float *)qg;
#pragma unroll
    for (uint32_t ks = 0; ks < 16u; ks++) {
        const uint32_t c = ks * 16u;
        const float *r0 = qbase + (uint64_t)(row0 + qrow) * DIM + c + qcol;
        const float *r1 = qbase + (uint64_t)(row0 + qrow + 8u) * DIM + c + qcol;
        qa[ks][0] = __float2half_rn(r0[0]);
        qa[ks][1] = __float2half_rn(r0[1]);
        qa[ks][2] = __float2half_rn(r1[0]);
        qa[ks][3] = __float2half_rn(r1[1]);
        qa[ks][4] = __float2half_rn(r0[8]);
        qa[ks][5] = __float2half_rn(r0[9]);
        qa[ks][6] = __float2half_rn(r1[8]);
        qa[ks][7] = __float2half_rn(r1[9]);
    }
    float sink = 0.0f;
    for (uint32_t rep = 0; rep < REPEAT; rep++) {
        for (uint32_t key = 0; key < n_keys; key += 8u) {
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (uint32_t ks = 0; ks < 16u; ks++) {
                __half b[4];
                const uint32_t kc = ks * 16u;
                const __half *bp = k_tile + (uint64_t)(key + (lane >> 2)) * DIM +
                                   kc + (lane & 3u) * 2u;
                b[0] = bp[0];
                b[1] = bp[1];
                b[2] = bp[8];
                b[3] = bp[9];
                hmma_m16n8k16(c, qa[ks], b);
            }
            sink += c[0] + c[1] + c[2] + c[3];
        }
    }
    if (sink == 12345.678f) out[0] = sink;
}

int main(void) {
    __half *k = nullptr, *q = nullptr;
    float *out = nullptr;
    const size_t tile = NKEYS * DIM * sizeof(__half);
    if (cudaMalloc(&k, tile) != cudaSuccess) return 1;
    if (cudaMalloc(&q, (size_t)16 * NBLK * DIM * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&out, 4096) != cudaSuccess) return 1;
    cudaMemset(k, 0x11, tile);
    /* Q in float, come nel motore (le proiezioni sono f32) */
    cudaMemset(q, 0, (size_t)16 * NBLK * DIM * sizeof(float));
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    /* MVARIANT 0: 16 warp per blocco (una riga ciascuno), 16 righe per blocco
     * MVARIANT 1: 1 warp per blocco, 16 righe nel frammento */
    const uint32_t threads = MVARIANT == 0 ? 512u : 32u;
    for (int pass = 0; pass < 2; pass++) {
#if MVARIANT == 0
        qk_scalar_kernel<<<NBLK, threads>>>(k, q, out, NKEYS);
#else
        qk_mma_kernel<<<NBLK, threads>>>(k, q, out, NKEYS);
#endif
        if (pass == 0) {
            if (cudaGetLastError() != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess) {
                printf("lancio fallito: %s\n", cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            continue;
        }
        cudaEventRecord(t0);
        for (int r = 0; r < 20; r++) {
#if MVARIANT == 0
            qk_scalar_kernel<<<NBLK, threads>>>(k, q, out, NKEYS);
#else
            qk_mma_kernel<<<NBLK, threads>>>(k, q, out, NKEYS);
#endif
        }
        cudaEventRecord(t1);
        if (cudaEventSynchronize(t1) != cudaSuccess) return 1;
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        const double per_call = ms / 20.0;
        /* 16 righe per blocco x 24 teste x NKEYS chiavi x REPEAT, per 16 layer */
        const double key_rows = (double)NBLK * NKEYS * 16.0 * REPEAT;
        const double per_kr = per_call * 1e6 / key_rows;
        printf("MVARIANT %d: %8.3f ms per lancio   %7.4f ns per (chiave,riga)"
               "   [release stamattina: 0,246 ns totali, di cui ~0,14 di dot]\n",
               MVARIANT, per_call, per_kr);
    }
    return 0;
}
