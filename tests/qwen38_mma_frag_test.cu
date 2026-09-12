/* Verifica numerica dei frammenti delle due MMA usate dall'attenzione GA.
 *
 * Nasce da un buco: lo spike (tests/qwen38_mma_qk_spike.cu) e' stato verificato
 * che compilasse e che cronometrasse, mai che calcolasse i numeri giusti.  Un
 * layout speculare produce punteggi correlati ma non uguali, cioe' esattamente
 * l'errore da 1-5 logits che la FA-2 mostrava.
 *
 *   nvcc -O3 -arch=sm_120a -o mma_frag_test tests/qwen38_mma_frag_test.cu
 *   ./mma_frag_test
 *
 * A e B sono riempiti con interi piccoli (esatti in f16, e i prodotti stanno in
 * f32 senza arrotondamenti), quindi l'MMA deve dare esattamente il prodotto
 * scalare: qualunque differenza e' una mappa di frammento sbagliata.
 *
 * Le mappe provate sono quelle del PTX ISA, le stesse scritte nei kernel:
 *   A (16x16 row-major)  a0a1 riga lane>>2, colonne (lane%4)*2+{0,1}
 *                        a2a3 riga+8,        a4a5 colonna+8, a6a7 entrambe +8
 *   B (16x8  col-major)  b0b1 righe (lane%4)*2+{0,1}, colonna lane>>2
 *                        b2b3 righe+8
 *   C (16x8  f32)        c0c1 riga lane>>2, colonne (lane%4)*2+{0,1}, c2c3 riga+8
 * e per m16n8k8 B ha una sola coppia (b0b1) e A due (a0a1, a2a3).
 */
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define AK 16
#define BK 16
#define BN 8

__device__ static __forceinline__ void mma_k16(float *c, const __half *a,
                                               const __half *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(*(const uint32_t *)&a[0]), "r"(*(const uint32_t *)&a[2]),
          "r"(*(const uint32_t *)&a[4]), "r"(*(const uint32_t *)&a[6]),
          "r"(*(const uint32_t *)&b[0]), "r"(*(const uint32_t *)&b[2]));
}

__device__ static __forceinline__ void mma_k8(float *c, const __half *a,
                                              const __half *b) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(*(const uint32_t *)&a[0]), "r"(*(const uint32_t *)&a[2]),
          "r"(*(const uint32_t *)&b[0]));
}

/* A[i][k] = 1 + 16i + k, B[k][j] = 1 + 8k + j: interi, esatti in f16, e ogni
 * valore identifica la sua cella. */
__global__ static void test_k16(const __half *a_g, const __half *b_g,
                                float *c_g) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t r0 = lane >> 2u;
    const uint32_t c0 = (lane & 3u) * 2u;
    __half a[8];
    a[0] = a_g[r0 * AK + c0 + 0u];
    a[1] = a_g[r0 * AK + c0 + 1u];
    a[2] = a_g[(r0 + 8u) * AK + c0 + 0u];
    a[3] = a_g[(r0 + 8u) * AK + c0 + 1u];
    a[4] = a_g[r0 * AK + c0 + 8u];
    a[5] = a_g[r0 * AK + c0 + 9u];
    a[6] = a_g[(r0 + 8u) * AK + c0 + 8u];
    a[7] = a_g[(r0 + 8u) * AK + c0 + 9u];
    __half b[4];
    b[0] = b_g[(c0 + 0u) * BN + r0];
    b[1] = b_g[(c0 + 1u) * BN + r0];
    b[2] = b_g[(c0 + 8u) * BN + r0];
    b[3] = b_g[(c0 + 9u) * BN + r0];
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mma_k16(c, a, b);
    c_g[r0 * BN + c0 + 0u] = c[0];
    c_g[r0 * BN + c0 + 1u] = c[1];
    c_g[(r0 + 8u) * BN + c0 + 0u] = c[2];
    c_g[(r0 + 8u) * BN + c0 + 1u] = c[3];
}

/* m16n8k8: A 16x8, B 8x8. */
__global__ static void test_k8(const __half *a_g, const __half *b_g, float *c_g) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t r0 = lane >> 2u;
    const uint32_t c0 = (lane & 3u) * 2u;
    __half a[4];
    a[0] = a_g[r0 * BN + c0 + 0u];
    a[1] = a_g[r0 * BN + c0 + 1u];
    a[2] = a_g[(r0 + 8u) * BN + c0 + 0u];
    a[3] = a_g[(r0 + 8u) * BN + c0 + 1u];
    __half b[2];
    b[0] = b_g[(c0 + 0u) * BN + r0];
    b[1] = b_g[(c0 + 1u) * BN + r0];
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mma_k8(c, a, b);
    c_g[r0 * BN + c0 + 0u] = c[0];
    c_g[r0 * BN + c0 + 1u] = c[1];
    c_g[(r0 + 8u) * BN + c0 + 0u] = c[2];
    c_g[(r0 + 8u) * BN + c0 + 1u] = c[3];
}

int main(void) {
    __half *a, *b;
    float *c, host_c[16 * BN], ref[16 * BN];
    if (cudaMalloc(&a, 16 * AK * sizeof(__half)) != cudaSuccess) return 1;
    if (cudaMalloc(&b, AK * BN * sizeof(__half)) != cudaSuccess) return 1;
    if (cudaMalloc(&c, 16 * BN * sizeof(float)) != cudaSuccess) return 1;
    __half host_a[16 * AK], host_b[AK * BN];
    for (int i = 0; i < 16; i++)
        for (int k = 0; k < AK; k++)
            host_a[i * AK + k] = __float2half_rn((float)(1 + 16 * i + k));
    for (int k = 0; k < AK; k++)
        for (int j = 0; j < BN; j++)
            host_b[k * BN + j] = __float2half_rn((float)(1 + 8 * k + j));

    /* --- m16n8k16 --- */
    cudaMemcpy(a, host_a, 16 * AK * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(b, host_b, AK * BN * sizeof(__half), cudaMemcpyHostToDevice);
    test_k16<<<1, 32>>>(a, b, c);
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        printf("lancio k16 fallito: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    cudaMemcpy(host_c, c, 16 * BN * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < BN; j++) {
            double sum = 0.0;
            for (int k = 0; k < AK; k++)
                sum += (double)(1 + 16 * i + k) * (double)(1 + 8 * k + j);
            ref[i * BN + j] = (float)sum;
        }
    int bad16 = 0;
    for (int i = 0; i < 16 * BN; i++)
        if (host_c[i] != ref[i]) bad16++;
    printf("m16n8k16: %d valori su %d sbagliati\n", bad16, 16 * BN);
    if (bad16) {
        printf("  prima riga gpu:");
        for (int j = 0; j < BN; j++) printf(" %10.1f", host_c[j]);
        printf("\n  prima riga ref:");
        for (int j = 0; j < BN; j++) printf(" %10.1f", ref[j]);
        printf("\n");
    }

    /* --- m16n8k8 --- */
    /* A has to be repacked to a stride of BN: the k16 pass above filled it with a
     * stride of AK, and reading it with the wrong stride was this test's own bug
     * the first time it ran. */
    {
        __half a8[16 * BN];
        for (int i = 0; i < 16; i++)
            for (int k = 0; k < BN; k++) a8[i * BN + k] = host_a[i * AK + k];
        cudaMemcpy(a, a8, sizeof(a8), cudaMemcpyHostToDevice);
    }
    for (int k = 0; k < AK; k++)
        for (int j = 0; j < BN; j++)
            host_b[k * BN + j] = __float2half_rn((float)(1 + 8 * k + j));
    cudaMemcpy(b, host_b, AK * BN * sizeof(__half), cudaMemcpyHostToDevice);
    test_k8<<<1, 32>>>(a, b, c);
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        printf("lancio k8 fallito: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    cudaMemcpy(host_c, c, 16 * BN * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < BN; j++) {
            double sum = 0.0;
            for (int k = 0; k < BN; k++)
                sum += (double)(1 + 16 * i + k) * (double)(1 + 8 * k + j);
            ref[i * BN + j] = (float)sum;
        }
    int bad8 = 0;
    for (int i = 0; i < 16 * BN; i++)
        if (host_c[i] != ref[i]) bad8++;
    printf("m16n8k8 : %d valori su %d sbagliati\n", bad8, 16 * BN);
    if (bad8) {
        printf("  matrice completa (gpu / ref), righe 0..15:\n");
        for (int i = 0; i < 16; i++) {
            printf("   r%-2d gpu:", i);
            for (int j = 0; j < BN; j++) printf(" %8.1f", host_c[i * BN + j]);
            printf("\n       ref:");
            for (int j = 0; j < BN; j++) printf(" %8.1f", ref[i * BN + j]);
            printf("%s\n", host_c[i*BN] == ref[i*BN] ? "" : "   <-- diversa");
        }
    }
    printf("%s\n", (bad16 == 0 && bad8 == 0) ? "FRAMMENTI CORRETTI"
                                             : "FRAMMENTI SBAGLIATI");
    return (bad16 == 0 && bad8 == 0) ? 0 : 1;
}
