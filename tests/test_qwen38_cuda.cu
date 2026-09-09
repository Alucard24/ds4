/* Mixed-IQ CUDA MMVQ smoke against ggml CPU dequantization.
 * Test-only dlopen dependency; ds4 itself never links llama.cpp/ggml. */
#include "ds4_mmq.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int ds4_cuda_q8_fold_take_q81(
        const void *, uint64_t, const void **q81) {
    if (q81) *q81 = nullptr;
    return 0;
}

struct quant_desc { uint32_t type, bytes; const char *name; };
static const quant_desc types[] = {
    {10,84,"q2_K"}, {12,144,"q4_K"}, {16,66,"iq2_xxs"},
    {17,74,"iq2_xs"}, {18,98,"iq3_xxs"}, {21,110,"iq3_s"},
    {22,82,"iq2_s"}, {23,136,"iq4_xs"}, {29,56,"iq1_m"},
};
static uint32_t rng = 0x5f3759dfu;
static uint32_t next_u32() { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }
static void die(const char *s) { std::fprintf(stderr, "%s\n", s); std::exit(1); }
static void cuda_check(cudaError_t e, const char *s) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", s, cudaGetErrorString(e)); std::exit(1); }
}
static uint16_t half_bits(float x) { __half h=__float2half_rn(x); uint16_t u; std::memcpy(&u,&h,2); return u; }
static void make_finite_block(uint8_t *b, uint32_t type) {
    const uint16_t d=half_bits(0.0005f+(next_u32()%1000)*0.000001f);
    if (type==29) {
        uint16_t s[4]; std::memcpy(s,b+48,8);
        for (int i=0;i<4;i++) s[i]=(s[i]&0x0fffu)|(((d>>(4*i))&15u)<<12);
        std::memcpy(b+48,s,8);
    } else if (type==10) {
        const uint16_t dm=half_bits(0.0004f);
        std::memcpy(b+80,&d,2); std::memcpy(b+82,&dm,2);
    } else {
        std::memcpy(b,&d,2);
        if (type==12) { uint16_t dm=half_bits(0.0004f); std::memcpy(b+2,&dm,2); }
    }
}
int main(int argc,char **argv) {
    if (argc!=2) die("usage: test_qwen38_cuda /path/to/libggml-cpu.so");
    void *lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL); if(!lib) die(dlerror());
    cuda_check(cudaSetDevice(0),"set device"); if(ds4_mmq_init(0)!=0) die("mmq init");
    constexpr int K=5120, M=64, N=1;
    std::vector<float> x(K), xq(K), ref(M), got(M), row(K);
    for(int pattern=0;pattern<2;pattern++) {
        for(int i=0;i<K;i++) x[i]=pattern==0 ? 1.0f :
            std::sin(i*0.137f)*1.7f+std::cos(i*0.019f)*0.3f;
        for(int ib=0;ib<K/32;ib++) {
            float amax=0; for(int j=0;j<32;j++) amax=std::fmax(amax,std::fabs(x[ib*32+j]));
            float d=__half2float(__float2half_rn(amax/127));
            for(int j=0;j<32;j++) xq[ib*32+j]=d*std::round(x[ib*32+j]/d);
        }
        for(const auto &t:types) {
            char sym[96]; std::snprintf(sym,sizeof(sym),"dequantize_row_%s",t.name);
            using deq_fn=void(*)(const void*,float*,int64_t);
            deq_fn deq=(deq_fn)dlsym(lib,sym); if(!deq) die(sym);
            const size_t row_bytes=(K/256)*t.bytes;
            std::vector<uint8_t> w((size_t)M*row_bytes);
            for(int r=0;r<M;r++) for(int ib=0;ib<K/256;ib++) {
                uint8_t *b=w.data()+(size_t)r*row_bytes+(size_t)ib*t.bytes;
                for(uint32_t j=0;j<t.bytes;j++) b[j]=(uint8_t)next_u32();
                make_finite_block(b,t.type);
            }
            for(int r=0;r<M;r++) {
                deq(w.data()+(size_t)r*row_bytes,row.data(),K);
                double sum=0; for(int i=0;i<K;i++) sum+=(double)row[i]*xq[i]; ref[r]=(float)sum;
            }
            void *dw=nullptr; float *dx=nullptr,*dy=nullptr;
            cuda_check(cudaMalloc(&dw,w.size()),"alloc W"); cuda_check(cudaMalloc(&dx,K*4),"alloc X");
            cuda_check(cudaMalloc(&dy,M*4),"alloc Y"); cuda_check(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice),"copy W");
            cuda_check(cudaMemcpy(dx,x.data(),K*4,cudaMemcpyHostToDevice),"copy X");
            if(ds4_mmq_quant_dense_vec(dw,t.type,dx,dy,M,N,K,0)!=0) die(t.name);
            cuda_check(cudaDeviceSynchronize(),"sync"); cuda_check(cudaMemcpy(got.data(),dy,M*4,cudaMemcpyDeviceToHost),"copy Y");
            double worst=0, mean=0;
            for(int r=0;r<M;r++) { double e=std::fabs(got[r]-ref[r]); worst=std::fmax(worst,e); mean+=e/M;
                const double limit=pattern==0 ? 5e-4 : 0.25+0.02*std::fabs(ref[r]);
                if(!std::isfinite(got[r]) || !std::isfinite(ref[r]) || e>limit) {
                    std::fprintf(stderr,"%s/%s row %d got %.8g ref %.8g err %.8g limit %.8g\n",
                        t.name,pattern?"varied":"constant",r,got[r],ref[r],e,limit); return 1; }
            }
            std::printf("%s/%s max_abs %.7g mean_abs %.7g PASS\n",
                t.name,pattern?"varied":"constant",worst,mean);
            if (t.type==22 && pattern==0) {
                float *drow=nullptr; cuda_check(cudaMalloc(&drow,K*4),"alloc embedding row");
                if(ds4_mmq_iq2_s_get_row(dw,drow,13,K,0)!=0) die("iq2_s get row");
                cuda_check(cudaDeviceSynchronize(),"get row sync");
                cuda_check(cudaMemcpy(got.data(),drow,M*4,cudaMemcpyDeviceToHost),"get row prefix");
                deq(w.data()+13*row_bytes,row.data(),K);
                for(int i=0;i<M;i++) if(got[i]!=row[i]) die("iq2_s get row mismatch");
                cudaFree(drow); std::puts("iq2_s/get-row prefix PASS");
            }
            cudaFree(dy); cudaFree(dx); cudaFree(dw);
        }
    }
    constexpr int BN=16;
    std::vector<float> bx((size_t)BN*K), bxq((size_t)BN*K);
    std::vector<float> bref((size_t)BN*M), bgot((size_t)BN*M);
    for(int n=0;n<BN;n++) for(int i=0;i<K;i++)
        bx[(size_t)n*K+i]=std::sin((i+17*n)*0.071f)*1.3f+
                          std::cos((i-11*n)*0.023f)*0.4f;
    for(int n=0;n<BN;n++) for(int ib=0;ib<K/32;ib++) {
        float amax=0; for(int j=0;j<32;j++)
            amax=std::fmax(amax,std::fabs(bx[(size_t)n*K+ib*32+j]));
        float d=__half2float(__float2half_rn(amax/127));
        for(int j=0;j<32;j++) bxq[(size_t)n*K+ib*32+j]=
            d*std::round(bx[(size_t)n*K+ib*32+j]/d);
    }
    for(const auto &t:types) {
        if(t.type==29) continue; /* IQ1_M has MMVQ only and is row-sliced by ds4. */
        char sym[96]; std::snprintf(sym,sizeof(sym),"dequantize_row_%s",t.name);
        using deq_fn=void(*)(const void*,float*,int64_t);
        deq_fn deq=(deq_fn)dlsym(lib,sym); if(!deq) die(sym);
        const size_t row_bytes=(K/256)*t.bytes;
        std::vector<uint8_t> w((size_t)M*row_bytes);
        for(int r=0;r<M;r++) for(int ib=0;ib<K/256;ib++) {
            uint8_t *b=w.data()+(size_t)r*row_bytes+(size_t)ib*t.bytes;
            for(uint32_t j=0;j<t.bytes;j++) b[j]=(uint8_t)next_u32();
            make_finite_block(b,t.type);
        }
        for(int r=0;r<M;r++) {
            deq(w.data()+(size_t)r*row_bytes,row.data(),K);
            for(int n=0;n<BN;n++) {
                double sum=0; for(int i=0;i<K;i++)
                    sum+=(double)row[i]*bxq[(size_t)n*K+i];
                bref[(size_t)n*M+r]=(float)sum;
            }
        }
        void *dw=nullptr; float *dx=nullptr,*dy=nullptr;
        cuda_check(cudaMalloc(&dw,w.size()),"batch alloc W");
        cuda_check(cudaMalloc(&dx,(size_t)BN*K*4),"batch alloc X");
        cuda_check(cudaMalloc(&dy,(size_t)BN*M*4),"batch alloc Y");
        cuda_check(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice),"batch copy W");
        cuda_check(cudaMemcpy(dx,bx.data(),(size_t)BN*K*4,cudaMemcpyHostToDevice),"batch copy X");
        if(ds4_mmq_quant_dense(dw,t.type,dx,dy,M,BN,K,0)!=0) die(t.name);
        cuda_check(cudaDeviceSynchronize(),"batch sync");
        cuda_check(cudaMemcpy(bgot.data(),dy,(size_t)BN*M*4,cudaMemcpyDeviceToHost),"batch copy Y");
        double worst=0;
        for(size_t i=0;i<bgot.size();i++) {
            const double e=std::fabs((double)bgot[i]-bref[i]);
            worst=std::fmax(worst,e);
            if(!std::isfinite(bgot[i]) || !std::isfinite(bref[i]) ||
               e>0.35+0.025*std::fabs(bref[i])) die("batch MMQ mismatch");
        }
        std::printf("%s/batch16 max_abs %.7g PASS\n",t.name,worst);
        cudaFree(dy); cudaFree(dx); cudaFree(dw);
    }
    dlclose(lib); return 0;
}
