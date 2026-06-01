#include "search/cpu_fine/cpu_fine.h"

namespace ivftensor {
namespace cpu_fine {

CpuFineKernelFn dispatch(CpuFineVariant v) {
    switch (v) {
        case CPU_FINE_V0: return &cpu_fine_kernel_v0;
        case CPU_FINE_V1: return &cpu_fine_kernel_v1;
        case CPU_FINE_V1_TOUCHED: return &cpu_fine_kernel_v1_touched;
        case CPU_FINE_V2: return &cpu_fine_kernel_v2;
        case CPU_FINE_V2_TOUCHED: return &cpu_fine_kernel_v2_touched;
        case CPU_FINE_V3: return &cpu_fine_kernel_v3;
        case CPU_FINE_V3_TOUCHED: return &cpu_fine_kernel_v3_touched;
        case CPU_FINE_V4: return &cpu_fine_kernel_v4;
        case CPU_FINE_V5: return &cpu_fine_kernel_v5;
        case CPU_FINE_V6: return &cpu_fine_kernel_v6;
        case CPU_FINE_V3_U8: return nullptr;         /* independent API; see cpu_fine_u8.h */
        case CPU_FINE_V3_U8_TOUCHED: return nullptr; /* independent API; see cpu_fine_u8.h */
        case CPU_FINE_PQ_RESID: return nullptr;      /* independent API; see cpu_fine_pq_resid.h */
        default: return nullptr;
    }
}

const char* variant_name(CpuFineVariant v) {
    switch (v) {
        case CPU_FINE_V0: return "V0_scalar_O0";
        case CPU_FINE_V1: return "V1_scalar_O3_march_native";
        case CPU_FINE_V1_TOUCHED: return "V1_touched_scalar_O3_march_native";
        case CPU_FINE_V2: return "V2_avx512_fma_4acc";
        case CPU_FINE_V2_TOUCHED: return "V2_touched_avx512_fma_4acc";
        case CPU_FINE_V3: return "V3_avx512_query_tile_Q4";
        case CPU_FINE_V3_TOUCHED: return "V3_touched_avx512_query_tile_Q4";
        case CPU_FINE_V4: return "V4_v3_plus_prefetch";
        case CPU_FINE_V5: return "V5_aosoa16_per_query";
        case CPU_FINE_V6: return "V6_cluster_outer_AoSoA16_Q4";
        case CPU_FINE_V3_U8: return "V3_u8_avx512_madd_epi16";
        case CPU_FINE_V3_U8_TOUCHED: return "V3_u8_touched_avx512_madd_epi16";
        case CPU_FINE_PQ_RESID: return "PQ_resid_M16_K256_rerank";
        default: return "UNKNOWN";
    }
}

}  // namespace cpu_fine
}  // namespace ivftensor
