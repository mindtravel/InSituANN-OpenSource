
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <condition_variable>
#include <cmath>
#include <future>
#include <mutex>
#include <limits>
#include <random>
#include <thread>
#include <unordered_set>
#include <utility>`n#include "search/cpu_fine/cpu_fine.h"
#define CUDA_CHECK(x) do { cudaError_t err__=(x); if(err__!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(err__)); exit(1);} } while(0)
constexpr int DIM=128, M=16, K=256, DSUB=8, RERANK=512, LOCAL_K=256, TPB=256, CHUNK=2048, GROUP_CHUNKS=8;
constexpr int DELTA_TOPK=16, COMBINED_CAND=RERANK+DELTA_TOPK;
static int64_t gN = 1000000000LL;
struct ChunkDesc { int qi, cid, start, len; };
struct GroupDesc { int qi, chunk_start, nchunks; };
static double wall_now_ms(){
  return std::chrono::duration<double,std::milli>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
static double percentile_ms(std::vector<double> xs,double p){
  if(xs.empty()) return 0.0;
  std::sort(xs.begin(),xs.end());
  double pos=(p/100.0)*(double)(xs.size()-1);
  size_t lo=(size_t)std::floor(pos), hi=(size_t)std::ceil(pos);
  if(lo==hi) return xs[lo];
  double t=pos-(double)lo;
  return xs[lo]*(1.0-t)+xs[hi]*t;
}
static uint64_t splitmix64(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
static bool is_deleted_id(int id,double ratio){if(ratio<=0.0||id<0)return false;if(ratio>=1.0)return true;long double th=ratio*(long double)std::numeric_limits<uint64_t>::max();return (long double)splitmix64((uint64_t)id)<th;}
static std::vector<float> read_centroids(const std::string& path,int nlist){std::ifstream f(path,std::ios::binary);if(!f){perror(path.c_str());exit(1);}int32_t nl=0,dim=0;f.read((char*)&nl,4);f.read((char*)&dim,4);if(nl!=nlist||dim!=DIM){std::cerr<<"bad centroids\n";exit(1);}std::vector<float> v((size_t)nlist*dim);f.read((char*)v.data(),v.size()*4);return v;}
static std::vector<int32_t> read_assign(const std::string& path){std::ifstream f(path,std::ios::binary);if(!f){perror(path.c_str());exit(1);}int64_t n=0;f.read((char*)&n,8);if(n!=gN){std::cerr<<"bad assign "<<n<<" expected "<<gN<<"\n";exit(1);}std::vector<int32_t>a(n);f.read((char*)a.data(),a.size()*4);return a;}
static std::vector<float> read_codebook(const std::string& path){std::ifstream f(path,std::ios::binary);if(!f){perror(path.c_str());exit(1);}int32_t m=0,k=0,dsub=0,pad=0;f.read((char*)&m,4);f.read((char*)&k,4);f.read((char*)&dsub,4);f.read((char*)&pad,4);if(m!=M||k!=K||dsub!=DSUB){std::cerr<<"bad cb\n";exit(1);}std::vector<float>v((size_t)M*K*DSUB);f.read((char*)v.data(),v.size()*4);return v;}
static std::vector<uint8_t> read_codes(const std::string& path){std::ifstream f(path,std::ios::binary);if(!f){perror(path.c_str());exit(1);}int64_t n=0;int32_t m=0,pad=0;f.read((char*)&n,8);f.read((char*)&m,4);f.read((char*)&pad,4);if(n!=gN||m!=M){std::cerr<<"bad codes "<<n<<" m="<<m<<" expected "<<gN<<"\n";exit(1);}std::vector<uint8_t>v((size_t)n*M);f.read((char*)v.data(),v.size());return v;}
static std::vector<float> read_queries(const std::string& path,int bs){std::ifstream f(path,std::ios::binary);if(!f){perror(path.c_str());exit(1);}int32_t nq=0,dim=0;f.read((char*)&nq,4);f.read((char*)&dim,4);if(dim!=DIM||nq<bs){std::cerr<<"bad query\n";exit(1);}std::vector<uint8_t>u((size_t)bs*dim);f.read((char*)u.data(),u.size());std::vector<float>q(u.size());for(size_t i=0;i<u.size();++i)q[i]=(float)u[i];return q;}
static std::vector<int> counts_from_assign(const std::vector<int32_t>& a,int nlist){std::vector<int>c(nlist);for(int x:a)c[x]++;return c;} static std::vector<long long> offsets_from_counts(const std::vector<int>& c){std::vector<long long>o(c.size());for(size_t i=1;i<c.size();++i)o[i]=o[i-1]+c[i-1];return o;}
__global__ void coarse_dist_kernel(const float*q,const float*cent,float*dists,int bs,int nlist){int idx=blockIdx.x*blockDim.x+threadIdx.x,total=bs*nlist;if(idx>=total)return;int qi=idx/nlist,cid=idx-qi*nlist;float d=0;const float*qq=q+(size_t)qi*DIM;const float*cc=cent+(size_t)cid*DIM;for(int j=0;j<DIM;j++){float x=qq[j]-cc[j];d+=x*x;}dists[idx]=d;}
__global__ void coarse_select_exact_kernel(const float*dists,int*out_ids,float*out_dist,int bs,int nlist,int nprobe){int qi=blockIdx.x;if(qi>=bs||threadIdx.x)return;extern __shared__ unsigned char sm[];float*bd=(float*)sm;int*bi=(int*)(bd+nprobe);for(int k=0;k<nprobe;k++){bd[k]=1e30f;bi[k]=-1;}const float*row=dists+(size_t)qi*nlist;for(int c=0;c<nlist;c++){float d=row[c];int worst=0;float wd=bd[0];for(int k=1;k<nprobe;k++)if(bd[k]>wd){wd=bd[k];worst=k;}if(d<wd){bd[worst]=d;bi[worst]=c;}}for(int a=0;a<nprobe;a++)for(int b=a+1;b<nprobe;b++)if(bd[b]<bd[a]){float td=bd[a];bd[a]=bd[b];bd[b]=td;int ti=bi[a];bi[a]=bi[b];bi[b]=ti;}for(int k=0;k<nprobe;k++){out_ids[qi*nprobe+k]=bi[k];out_dist[qi*nprobe+k]=bd[k];}}
__device__ inline void sort_pair(float&a,int&ia,float&b,int&ib,bool asc){bool sw=asc?(a>b):(a<b);if(sw){float td=a;a=b;b=td;int ti=ia;ia=ib;ib=ti;}}
__global__ void chunk_top256_select_kernel(const float*q,const float*cent,const float*cb,const uint8_t*codes,const long long*offsets,const ChunkDesc*desc,int nchunks,float*outd,int*outi){
  __shared__ float lut[M*K];
  __shared__ float sd[LOCAL_K];
  __shared__ int si[LOCAL_K];
  int bid=blockIdx.x; if(bid>=nchunks)return;
  ChunkDesc cd=desc[bid];
  const float*qq=q+(size_t)cd.qi*DIM;
  const float*cc=cent+(size_t)cd.cid*DIM;
  for(int e=threadIdx.x;e<M*K;e+=blockDim.x){
    int m=e/K,kk=e-m*K; const float*cw=cb+((m*K+kk)*DSUB); float s=0;
    for(int j=0;j<DSUB;j++){float r=qq[m*DSUB+j]-cc[m*DSUB+j]; float x=r-cw[j]; s+=x*x;}
    lut[e]=s;
  }
  __syncthreads();
  long long off=offsets[cd.cid]+cd.start;
  float best=1e30f; int besti=-1;
  for(int x=threadIdx.x;x<CHUNK;x+=blockDim.x){
    if(x<cd.len){
      int idx=(int)(off+x); const uint8_t*code=codes+(size_t)idx*M; float d=0;
      for(int m=0;m<M;m++) d+=lut[m*K+code[m]];
      if(d<best){best=d; besti=idx;}
    }
  }
  sd[threadIdx.x]=best; si[threadIdx.x]=besti;
  __syncthreads();
  for(int k=2;k<=LOCAL_K;k<<=1){
    for(int j=k>>1;j>0;j>>=1){
      for(int idx=threadIdx.x;idx<LOCAL_K;idx+=blockDim.x){
        int ixj=idx^j;
        if(ixj>idx){bool asc=((idx&k)==0); float a=sd[idx],b=sd[ixj]; int ia=si[idx],ib=si[ixj]; sort_pair(a,ia,b,ib,asc); sd[idx]=a; si[idx]=ia; sd[ixj]=b; si[ixj]=ib;}
      }
      __syncthreads();
    }
  }
  for(int k=threadIdx.x;k<LOCAL_K;k+=blockDim.x){outd[(size_t)bid*LOCAL_K+k]=sd[k]; outi[(size_t)bid*LOCAL_K+k]=si[k];}
}
__global__ void group_merge_local_kernel(const GroupDesc*gdesc,int ngroups,const float*chunkd,const int*chunki,float*outd,int*outi){
  extern __shared__ unsigned char sm[];
  float*sd=(float*)sm; int*si=(int*)(sd+GROUP_CHUNKS*LOCAL_K);
  int gid=blockIdx.x; if(gid>=ngroups)return;
  GroupDesc gd=gdesc[gid]; int cand=gd.nchunks*LOCAL_K;
  for(int x=threadIdx.x;x<GROUP_CHUNKS*LOCAL_K;x+=blockDim.x){
    if(x<cand){int ch=x/LOCAL_K,k=x-ch*LOCAL_K; size_t pos=(size_t)(gd.chunk_start+ch)*LOCAL_K+k; sd[x]=chunkd[pos]; si[x]=chunki[pos];}
    else{sd[x]=1e30f; si[x]=-1;}
  }
  __syncthreads();
  for(int k=2;k<=GROUP_CHUNKS*LOCAL_K;k<<=1){
    for(int j=k>>1;j>0;j>>=1){
      for(int idx=threadIdx.x;idx<GROUP_CHUNKS*LOCAL_K;idx+=blockDim.x){
        int ixj=idx^j;
        if(ixj>idx){bool asc=((idx&k)==0); float a=sd[idx],b=sd[ixj]; int ia=si[idx],ib=si[ixj]; sort_pair(a,ia,b,ib,asc); sd[idx]=a; si[idx]=ia; sd[ixj]=b; si[ixj]=ib;}
      }
      __syncthreads();
    }
  }
  for(int k=threadIdx.x;k<RERANK;k+=blockDim.x){
    if(k<cand){outd[(size_t)gid*RERANK+k]=sd[k]; outi[(size_t)gid*RERANK+k]=si[k];}
    else{outd[(size_t)gid*RERANK+k]=1e30f; outi[(size_t)gid*RERANK+k]=-1;}
  }
}
__global__ void group_merge_kernel(const GroupDesc*gdesc,int ngroups,const float*chunkd,const int*chunki,float*outd,int*outi){extern __shared__ unsigned char sm[];float*sd=(float*)sm;int*si=(int*)(sd+GROUP_CHUNKS*RERANK);int gid=blockIdx.x;if(gid>=ngroups)return;GroupDesc gd=gdesc[gid];int cand=gd.nchunks*RERANK;for(int x=threadIdx.x;x<GROUP_CHUNKS*RERANK;x+=blockDim.x){if(x<cand){int ch=x/RERANK,k=x-ch*RERANK;size_t pos=(size_t)(gd.chunk_start+ch)*RERANK+k;sd[x]=chunkd[pos];si[x]=chunki[pos];}else{sd[x]=1e30f;si[x]=-1;}}__syncthreads();for(int k=2;k<=GROUP_CHUNKS*RERANK;k<<=1){for(int j=k>>1;j>0;j>>=1){for(int idx=threadIdx.x;idx<GROUP_CHUNKS*RERANK;idx+=blockDim.x){int ixj=idx^j;if(ixj>idx){bool asc=((idx&k)==0);float a=sd[idx],b=sd[ixj];int ia=si[idx],ib=si[ixj];sort_pair(a,ia,b,ib,asc);sd[idx]=a;si[idx]=ia;sd[ixj]=b;si[ixj]=ib;}}__syncthreads();}}for(int k=threadIdx.x;k<RERANK;k+=blockDim.x){outd[(size_t)gid*RERANK+k]=sd[k];outi[(size_t)gid*RERANK+k]=si[k];}}
__global__ void final_merge_kernel(const int* qgstart,const int* qgcount,const float*gd,const int*gi,float*outd,int*outi,int bs){int qi=blockIdx.x;if(qi>=bs||threadIdx.x)return;float topd[RERANK];int topi[RERANK];for(int k=0;k<RERANK;k++){topd[k]=1e30f;topi[k]=-1;}int st=qgstart[qi],cnt=qgcount[qi];for(int g=0;g<cnt;g++)for(int k=0;k<RERANK;k++){float d=gd[(size_t)(st+g)*RERANK+k];int idx=gi[(size_t)(st+g)*RERANK+k];int worst=0;float wd=topd[0];for(int t=1;t<RERANK;t++)if(topd[t]>wd){wd=topd[t];worst=t;}if(d<wd){topd[worst]=d;topi[worst]=idx;}}for(int a=0;a<RERANK;a++)for(int b=a+1;b<RERANK;b++)if(topd[b]<topd[a]){float td=topd[a];topd[a]=topd[b];topd[b]=td;int ti=topi[a];topi[a]=topi[b];topi[b]=ti;}for(int k=0;k<RERANK;k++){outd[qi*RERANK+k]=topd[k];outi[qi*RERANK+k]=topi[k];}}
__global__ void fused_first_group_merge_kernel(
  const float*q,const float*cent,const float*cb,const uint8_t*codes,const long long*offsets,
  const ChunkDesc*desc,const GroupDesc*gdesc,int ngroups,float*outd,int*outi){
  extern __shared__ unsigned char sm[];
  float* sd=(float*)sm; int* si=(int*)(sd+GROUP_CHUNKS*LOCAL_K);
  __shared__ float lut[M*K];
  int gid=blockIdx.x; if(gid>=ngroups)return;
  GroupDesc gd=gdesc[gid];

  for(int ch=0; ch<gd.nchunks; ++ch){
    ChunkDesc cd=desc[gd.chunk_start+ch];
    const float* qq=q+(size_t)cd.qi*DIM;
    const float* cc=cent+(size_t)cd.cid*DIM;
    for(int e=threadIdx.x; e<M*K; e+=blockDim.x){
      int m=e/K,kk=e-m*K; const float*cw=cb+((m*K+kk)*DSUB); float s=0;
      for(int j=0;j<DSUB;j++){float r=qq[m*DSUB+j]-cc[m*DSUB+j]; float x=r-cw[j]; s+=x*x;}
      lut[e]=s;
    }
    __syncthreads();

    for(int lane=threadIdx.x; lane<LOCAL_K; lane+=blockDim.x){
      long long off=offsets[cd.cid]+cd.start;
      float best=1e30f; int besti=-1;
      for(int item=lane; item<CHUNK; item+=LOCAL_K){
        if(item<cd.len){
          int idx=(int)(off+item); const uint8_t*code=codes+(size_t)idx*M; float d=0;
          for(int m=0;m<M;m++) d+=lut[m*K+code[m]];
          if(d<best){best=d; besti=idx;}
        }
      }
      int x=ch*LOCAL_K+lane;
      sd[x]=best; si[x]=besti;
    }
    __syncthreads();
  }
  for(int x=threadIdx.x+gd.nchunks*LOCAL_K; x<GROUP_CHUNKS*LOCAL_K; x+=blockDim.x){
    sd[x]=1e30f; si[x]=-1;
  }
  __syncthreads();

  for(int k=2;k<=GROUP_CHUNKS*LOCAL_K;k<<=1){
    for(int j=k>>1;j>0;j>>=1){
      for(int idx=threadIdx.x;idx<GROUP_CHUNKS*LOCAL_K;idx+=blockDim.x){
        int ixj=idx^j;
        if(ixj>idx){bool asc=((idx&k)==0); float a=sd[idx],b=sd[ixj]; int ia=si[idx],ib=si[ixj]; sort_pair(a,ia,b,ib,asc); sd[idx]=a; si[idx]=ia; sd[ixj]=b; si[ixj]=ib;}
      }
      __syncthreads();
    }
  }
  int cand=gd.nchunks*LOCAL_K;
  for(int k=threadIdx.x;k<RERANK;k+=blockDim.x){
    if(k<cand){outd[(size_t)gid*RERANK+k]=sd[k]; outi[(size_t)gid*RERANK+k]=si[k];}
    else{outd[(size_t)gid*RERANK+k]=1e30f; outi[(size_t)gid*RERANK+k]=-1;}
  }
}
__global__ void copy_final_groups_kernel(const int* qgstart,const float*gd,const int*gi,float*outd,int*outi,int bs){int qi=blockIdx.x;if(qi>=bs)return;int g=qgstart[qi];for(int k=threadIdx.x;k<RERANK;k+=blockDim.x){outd[(size_t)qi*RERANK+k]=gd[(size_t)g*RERANK+k];outi[(size_t)qi*RERANK+k]=gi[(size_t)g*RERANK+k];}}

__global__ void map_delta_candidate_ids_kernel(int* ids,int n,const int* id_map){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=n) return;
  int local=ids[idx];
  if(local>=0) ids[idx]=id_map[local];
}

__global__ void pair_chunk_count_kernel(const int* hcids,const int* counts,int total_pairs,int* pair_counts){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=total_pairs)return;
  int cid=hcids[idx];
  int cnt=counts[cid];
  pair_counts[idx]=(cnt+CHUNK-1)/CHUNK;
}
__global__ void query_chunk_count_kernel(const int* pair_counts,int bs,int nprobe,int* q_counts){
  int qi=blockIdx.x;
  if(qi>=bs)return;
  int sum=0;
  for(int p=threadIdx.x;p<nprobe;p+=blockDim.x) sum+=pair_counts[(size_t)qi*nprobe+p];
  __shared__ int partial[256];
  partial[threadIdx.x]=sum;
  __syncthreads();
  for(int stride=blockDim.x/2;stride>0;stride>>=1){
    if(threadIdx.x<stride) partial[threadIdx.x]+=partial[threadIdx.x+stride];
    __syncthreads();
  }
  if(threadIdx.x==0) q_counts[qi]=partial[0];
}
__global__ void emit_chunk_desc_kernel(const int* hcids,const int* counts,const int* pair_offsets,int total_pairs,int nprobe,ChunkDesc* desc){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=total_pairs)return;
  int qi=idx/nprobe;
  int cid=hcids[idx];
  int cnt=counts[cid];
  int out=pair_offsets[idx];
  for(int st=0,k=0; st<cnt; st+=CHUNK,++k){
    int rem=cnt-st;
    int len=rem<CHUNK?rem:CHUNK;
    desc[out+k]={qi,cid,st,len};
  }
}
__global__ void group_count_kernel(const int* cur_count,int bs,int* group_count){
  int qi=blockIdx.x*blockDim.x+threadIdx.x;
  if(qi>=bs)return;
  int c=cur_count[qi];
  group_count[qi]=(c+GROUP_CHUNKS-1)/GROUP_CHUNKS;
}
__global__ void emit_group_desc_kernel(const int* cur_start,const int* cur_count,const int* group_start,int bs,GroupDesc* gdesc){
  int qi=blockIdx.x;
  if(qi>=bs)return;
  int c=cur_count[qi];
  int gs=group_start[qi];
  int base=cur_start[qi];
  int ng=(c+GROUP_CHUNKS-1)/GROUP_CHUNKS;
  for(int j=threadIdx.x;j<ng;j+=blockDim.x){
    int remain=c-j*GROUP_CHUNKS;
    int len=remain<GROUP_CHUNKS?remain:GROUP_CHUNKS;
    gdesc[gs+j]={qi,base+j*GROUP_CHUNKS,len};
  }
}
static void device_exclusive_scan_int_ws(const int* in,int* out,int n,void** tmp,size_t* tmp_bytes){
  void* null_tmp=nullptr;
  size_t need=0;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(null_tmp,need,in,out,n));
  if(need>*tmp_bytes){
    if(*tmp) CUDA_CHECK(cudaFree(*tmp));
    CUDA_CHECK(cudaMalloc(tmp,need));
    *tmp_bytes=need;
  }
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(*tmp,*tmp_bytes,in,out,n));
}
static double cpu_rerank_ms(const std::string& base,const std::vector<float>&q,const std::vector<int>&cand,int bs){std::ifstream f(base,std::ios::binary);if(!f){perror(base.c_str());exit(1);}int32_t n=0,dim=0;f.read((char*)&n,4);f.read((char*)&dim,4);std::vector<uint8_t>v(DIM);auto t0=std::chrono::high_resolution_clock::now();for(int qi=0;qi<bs;qi++){float best=1e30f;for(int k=0;k<RERANK;k++){int idx=cand[qi*RERANK+k];if(idx<0)continue;f.seekg(8LL+(long long)idx*DIM);f.read((char*)v.data(),DIM);float d=0;const float*qq=q.data()+(size_t)qi*DIM;for(int j=0;j<DIM;j++){float x=qq[j]-(float)v[j];d+=x*x;}if(d<best)best=d;}}auto t1=std::chrono::high_resolution_clock::now();return std::chrono::duration<double,std::milli>(t1-t0).count();}


static std::vector<uint8_t> read_base_u8_memory(const std::string& path){
  std::ifstream f(path, std::ios::binary);
  if(!f){perror(path.c_str()); exit(1);}
  int32_t n=0, dim=0; f.read((char*)&n,4); f.read((char*)&dim,4);
  if(n!=gN || dim!=DIM){std::cerr<<"bad base header "<<n<<" "<<dim<<"\n"; exit(1);}
  std::vector<uint8_t> base((size_t)n*dim);
  f.read((char*)base.data(), base.size());
  return base;
}
static std::vector<int> build_reorder_to_original_from_assign(const std::vector<int32_t>& assign, const std::vector<long long>& offsets){
  std::vector<long long> next = offsets;
  std::vector<int> reorder_to_original(assign.size(), -1);
  for(size_t orig=0; orig<assign.size(); ++orig){
    int cid = assign[orig];
    long long pos = next[cid]++;
    reorder_to_original[(size_t)pos] = (int)orig;
  }
  return reorder_to_original;
}
struct RecallResult { double ms; double checksum; double recall1; double recall10; double recall100; };
static volatile double g_rerank_checksum_sink = 0.0;
static std::vector<int> read_groundtruth100(const std::string& path, int bs){
  std::ifstream f(path, std::ios::binary);
  if(!f){perror(path.c_str()); exit(1);}
  int32_t nq=0, k=0; f.read((char*)&nq,4); f.read((char*)&k,4);
  if(nq<bs || k<100){std::cerr<<"bad gt header "<<nq<<" "<<k<<"\n"; exit(1);}
  std::vector<int> gt((size_t)bs*100);
  for(int qi=0; qi<bs; ++qi){
    f.read((char*)(gt.data()+(size_t)qi*100), 100*4);
  }
  return gt;
}
static RecallResult cpu_rerank_mem_recall_ms(const std::vector<uint8_t>& base,const std::vector<float>&q,const std::vector<int>&cand,const std::vector<int>&reorder_to_original,const std::vector<int>&gt,int bs){
  std::vector<int> final_idx((size_t)bs*100, -1);
  std::vector<float> final_dist((size_t)bs*100, 1e30f);
  double checksum = 0.0;
  auto t0=std::chrono::high_resolution_clock::now();
  #pragma omp parallel for schedule(static) reduction(+:checksum)
  for(int qi=0; qi<bs; ++qi){
    float topd[100]; int topi[100];
    for(int t=0;t<100;t++){topd[t]=1e30f; topi[t]=-1;}
    const float* qq=q.data()+(size_t)qi*DIM;

    std::pair<int,int> read_order[RERANK];
    int orig_by_pos[RERANK];
    float dist_by_pos[RERANK];
    int nvalid=0;
    for(int k=0;k<RERANK;k++){
      orig_by_pos[k] = -1;
      dist_by_pos[k] = 1e30f;
      int ridx=cand[(size_t)qi*RERANK+k]; if(ridx<0) continue;
      int idx=reorder_to_original[(size_t)ridx]; if(idx<0) continue;
      orig_by_pos[k] = idx;
      read_order[nvalid++] = {idx, k};
    }
    std::sort(read_order, read_order+nvalid,
              [](const std::pair<int,int>& a,const std::pair<int,int>& b){
                return a.first < b.first || (a.first == b.first && a.second < b.second);
              });

    for(int r=0;r<nvalid;r++){
#if defined(__GNUC__)
      constexpr int PREFETCH_DIST = 8;
      if(r+PREFETCH_DIST<nvalid){
        const uint8_t* pv=base.data()+(size_t)read_order[r+PREFETCH_DIST].first*DIM;
        __builtin_prefetch(pv, 0, 1);
      }
#endif
      int idx=read_order[r].first;
      int pos=read_order[r].second;
      const uint8_t* v=base.data()+(size_t)idx*DIM;
      float d=0.f;
      for(int j=0;j<DIM;j++){float x=qq[j]-(float)v[j]; d+=x*x;}
      dist_by_pos[pos]=d;
    }

    // Preserve the original candidate order for tie behavior; only the base reads are reordered.
    for(int k=0;k<RERANK;k++){
      int idx=orig_by_pos[k]; if(idx<0) continue;
      float d=dist_by_pos[k];
      int worst=0; float wd=topd[0];
      for(int t=1;t<100;t++) if(topd[t]>wd){wd=topd[t]; worst=t;}
      if(d<wd){topd[worst]=d; topi[worst]=idx;}
    }
    for(int a=0;a<100;a++) for(int b=a+1;b<100;b++) if(topd[b]<topd[a]){float td=topd[a]; topd[a]=topd[b]; topd[b]=td; int ti=topi[a]; topi[a]=topi[b]; topi[b]=ti;}
    for(int t=0;t<100;t++){final_idx[(size_t)qi*100+t]=topi[t]; final_dist[(size_t)qi*100+t]=topd[t]; checksum += topd[t] * (t+1);}
  }
  auto t1=std::chrono::high_resolution_clock::now();
  long long hit1=0, hit10=0, hit100=0;
  for(int qi=0; qi<bs; ++qi){
    if(final_idx[(size_t)qi*100] == gt[(size_t)qi*100]) hit1++;
    for(int g=0; g<10; ++g){
      int id=gt[(size_t)qi*100+g];
      for(int r=0;r<10;r++) if(final_idx[(size_t)qi*100+r]==id){hit10++; break;}
    }
    for(int g=0; g<100; ++g){
      int id=gt[(size_t)qi*100+g];
      for(int r=0;r<100;r++) if(final_idx[(size_t)qi*100+r]==id){hit100++; break;}
    }
  }
  g_rerank_checksum_sink = checksum;
  RecallResult rr;
  rr.ms = std::chrono::duration<double,std::milli>(t1-t0).count();
  rr.checksum = checksum;
  rr.recall1 = (double)hit1 / (double)bs;
  rr.recall10 = (double)hit10 / (double)(bs*10);
  rr.recall100 = (double)hit100 / (double)(bs*100);
  return rr;
}



static std::vector<uint8_t> read_base_u8bin_prefix_memory(const std::string& path, int64_t expected_n){
  std::ifstream f(path, std::ios::binary);
  if(!f){perror(path.c_str()); exit(1);}
  int32_t n32=0, dim=0; f.read((char*)&n32,4); f.read((char*)&dim,4);
  uint32_t un=(uint32_t)n32;
  int64_t n=(int64_t)un;
  if(n < expected_n || dim != DIM){std::cerr<<"bad u8bin header "<<n<<" "<<dim<<" expected_n="<<expected_n<<"\n"; exit(1);}
  std::vector<uint8_t> v((size_t)expected_n*DIM);
  f.read((char*)v.data(), v.size());
  return v;
}
static std::vector<float> read_query_u8bin_prefix(const std::string& path, int bs){
  std::ifstream f(path, std::ios::binary);
  if(!f){perror(path.c_str()); exit(1);}
  int32_t n=0, dim=0; f.read((char*)&n,4); f.read((char*)&dim,4);
  if(n < bs || dim != DIM){std::cerr<<"bad query header "<<n<<" "<<dim<<"\n"; exit(1);}
  std::vector<uint8_t> u((size_t)bs*DIM);
  f.read((char*)u.data(), u.size());
  std::vector<float> q(u.size());
  for(size_t i=0;i<u.size();++i) q[i]=(float)u[i];
  return q;
}
static std::vector<int> read_groundtruth100_anyk(const std::string& path, int bs){
  std::ifstream f(path, std::ios::binary);
  if(!f){perror(path.c_str()); exit(1);}
  int32_t nq=0, k=0; f.read((char*)&nq,4); f.read((char*)&k,4);
  if(nq<bs || k<100){std::cerr<<"bad gt header "<<nq<<" "<<k<<"\n"; exit(1);}
  std::vector<int> gt((size_t)bs*100);
  std::vector<int> row(k);
  for(int qi=0; qi<bs; ++qi){
    f.read((char*)row.data(), (size_t)k*4);
    for(int j=0;j<100;j++) gt[(size_t)qi*100+j]=row[j];
  }
  return gt;
}

struct DeltaPqGpuSegment {
  int active_n = 0;
  int nlist = 0;
  std::vector<int> counts;
  std::vector<long long> offsets;
  std::vector<uint8_t> codes;
  std::vector<int> ids;
  uint8_t* d_codes = nullptr;
  long long* d_offsets = nullptr;
  int* d_counts = nullptr;
  int* d_ids = nullptr;

  void upload() {
    CUDA_CHECK(cudaMalloc(&d_codes, codes.size()));
    CUDA_CHECK(cudaMalloc(&d_offsets, offsets.size() * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_counts, counts.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ids, ids.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_counts, counts.data(), counts.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
  }

  void release() {
    if(d_codes) cudaFree(d_codes);
    if(d_offsets) cudaFree(d_offsets);
    if(d_counts) cudaFree(d_counts);
    if(d_ids) cudaFree(d_ids);
    d_codes = nullptr;
    d_offsets = nullptr;
    d_counts = nullptr;
    d_ids = nullptr;
  }
};

static int delta_cluster_for_insert_id(int local_id, int nlist) {
  return (int)(((uint64_t)local_id * 11400714819323198485ull) % (uint64_t)nlist);
}

static std::vector<uint8_t> make_random_delta_u8(int max_delta_n, uint64_t seed) {
  std::vector<uint8_t> delta((size_t)max_delta_n * DIM);
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  for(size_t i=0;i<delta.size();++i) delta[i]=(uint8_t)byte_dist(rng);
  std::cerr<<"[DELTA] random-u8 generated "<<max_delta_n<<" vectors x "<<DIM<<" seed="<<seed<<"\n";
  return delta;
}

static uint8_t synthetic_delta_pq_code(int local_id, int m) {
  return (uint8_t)(splitmix64((uint64_t)local_id ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(m + 1))) & 255ULL);
}

static DeltaPqGpuSegment build_delta_pq_mutable_publish(int active_n, int nlist, int base_id) {
  DeltaPqGpuSegment seg;
  seg.active_n = active_n;
  seg.nlist = nlist;
  seg.counts.assign((size_t)nlist, 0);
  seg.offsets.assign((size_t)nlist, 0);
  std::vector<int> reserve_counts((size_t)nlist, 0);
  for(int i=0;i<active_n;i++) reserve_counts[(size_t)delta_cluster_for_insert_id(i,nlist)]++;

  auto t0=std::chrono::high_resolution_clock::now();
  std::vector<std::vector<int>> mutable_delta_lists((size_t)nlist);
  for(int c=0;c<nlist;c++) if(reserve_counts[(size_t)c]>0) mutable_delta_lists[(size_t)c].reserve((size_t)reserve_counts[(size_t)c]);
  for(int i=0;i<active_n;i++) mutable_delta_lists[(size_t)delta_cluster_for_insert_id(i,nlist)].push_back(i);
  auto t1=std::chrono::high_resolution_clock::now();

  int max_list=0; long long nonempty=0;
  for(int c=0;c<nlist;c++){
    int cnt=(int)mutable_delta_lists[(size_t)c].size();
    seg.counts[(size_t)c]=cnt;
    max_list=std::max(max_list,cnt);
    nonempty+=(cnt>0);
  }
  for(int c=1;c<nlist;c++) seg.offsets[(size_t)c]=seg.offsets[(size_t)c-1]+seg.counts[(size_t)c-1];
  seg.codes.assign((size_t)active_n * M, 0);
  seg.ids.assign((size_t)active_n, -1);
  for(int c=0;c<nlist;c++){
    long long dst=seg.offsets[(size_t)c];
    for(int local_id: mutable_delta_lists[(size_t)c]){
      uint8_t* out_code=seg.codes.data()+(size_t)dst*M;
      for(int m=0;m<M;m++) out_code[m]=synthetic_delta_pq_code(local_id,m);
      seg.ids[(size_t)dst]=base_id+local_id;
      ++dst;
    }
  }
  auto t2=std::chrono::high_resolution_clock::now();
  double append_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
  double publish_ms=std::chrono::duration<double,std::milli>(t2-t1).count();
  std::cerr<<"[DELTA-PQ-MUTABLE] active_n="<<active_n
           <<" append_ms="<<append_ms<<" publish_flatten_codes_ms="<<publish_ms
           <<" nonempty_lists="<<nonempty<<" max_list="<<max_list<<"\n";
  return seg;
}

static const DeltaPqGpuSegment* find_delta_pq_segment(const std::vector<DeltaPqGpuSegment>& segs, int active_n) {
  for(const auto& s: segs) if(s.active_n==active_n) return &s;
  return nullptr;
}

static RecallResult cpu_rerank_mem_recall_ms_update(const std::vector<uint8_t>& base,
                                                    const std::vector<uint8_t>& delta_u8,
                                                    const std::vector<float>& q,
                                                    const std::vector<int>& cand,
                                                    int cand_k,
                                                    const std::vector<int>& reorder_to_original,
                                                    const std::vector<int>& gt,
                                                    const std::unordered_set<int>* delete_protect,
                                                    int bs,
                                                    double delete_ratio) {
  std::vector<int> final_idx((size_t)bs*100, -1);
  double checksum=0.0;
  auto t0=std::chrono::high_resolution_clock::now();
  #pragma omp parallel for schedule(static) reduction(+:checksum)
  for(int qi=0; qi<bs; ++qi){
    float topd[100]; int topi[100];
    for(int t=0;t<100;t++){topd[t]=1e30f; topi[t]=-1;}
    const float* qq=q.data()+(size_t)qi*DIM;
    for(int k=0;k<cand_k;k++){
      int ridx=cand[(size_t)qi*cand_k+k];
      if(ridx<0) continue;
      const uint8_t* v=nullptr;
      int final_id=-1;
      if((int64_t)ridx < gN){
        if((size_t)ridx>=reorder_to_original.size()) continue;
        int orig=reorder_to_original[(size_t)ridx];
        bool protected_id = delete_protect && delete_protect->find(orig) != delete_protect->end();
        if(orig<0 || (is_deleted_id(orig,delete_ratio) && !protected_id)) continue;
        v=base.data()+(size_t)orig*DIM;
        final_id=orig;
      } else {
        int64_t local=(int64_t)ridx-gN;
        if(local<0 || (size_t)(local+1)*DIM>delta_u8.size()) continue;
        v=delta_u8.data()+(size_t)local*DIM;
        final_id=ridx;
      }
      float d=0.f;
      for(int j=0;j<DIM;j++){float x=qq[j]-(float)v[j]; d+=x*x;}
      int worst=0; float wd=topd[0];
      for(int t=1;t<100;t++) if(topd[t]>wd){wd=topd[t]; worst=t;}
      if(d<wd){topd[worst]=d; topi[worst]=final_id;}
    }
    for(int a=0;a<100;a++) for(int b=a+1;b<100;b++) if(topd[b]<topd[a]){float td=topd[a];topd[a]=topd[b];topd[b]=td;int ti=topi[a];topi[a]=topi[b];topi[b]=ti;}
    for(int t=0;t<100;t++){final_idx[(size_t)qi*100+t]=topi[t]; checksum+=topd[t]*(t+1);}
  }
  auto t1=std::chrono::high_resolution_clock::now();
  long long hit1=0,hit10=0,hit100=0;
  for(int qi=0; qi<bs; ++qi){
    if(final_idx[(size_t)qi*100] == gt[(size_t)qi*100]) hit1++;
    for(int g=0; g<10; ++g){
      int id=gt[(size_t)qi*100+g];
      for(int r=0;r<10;r++) if(final_idx[(size_t)qi*100+r]==id){hit10++; break;}
    }
    for(int g=0; g<100; ++g){
      int id=gt[(size_t)qi*100+g];
      for(int r=0;r<100;r++) if(final_idx[(size_t)qi*100+r]==id){hit100++; break;}
    }
  }
  RecallResult rr;
  rr.ms=std::chrono::duration<double,std::milli>(t1-t0).count();
  rr.checksum=checksum;
  rr.recall1=(double)hit1/(double)bs;
  rr.recall10=(double)hit10/(double)(bs*10);
  rr.recall100=(double)hit100/(double)(bs*100);
  return rr;
}
static double cpu_rerank_f32_recall_ms(const std::vector<float>& base,const std::vector<float>&q,const std::vector<int>&cand,const std::vector<int>&reorder_to_original,const std::vector<int>&gt,int bs, RecallResult* out){
  std::vector<int> final_idx((size_t)bs*100, -1);
  std::vector<float> final_dist((size_t)bs*100, 1e30f);
  double checksum = 0.0;
  auto t0=std::chrono::high_resolution_clock::now();
  #pragma omp parallel for schedule(static) reduction(+:checksum)
  for(int qi=0; qi<bs; ++qi){
    float topd[100]; int topi[100];
    for(int t=0;t<100;t++){topd[t]=1e30f; topi[t]=-1;}
    const float* qq=q.data()+(size_t)qi*DIM;
    std::pair<int,int> read_order[RERANK];
    int orig_by_pos[RERANK];
    float dist_by_pos[RERANK];
    int nvalid=0;
    for(int k=0;k<RERANK;k++){
      orig_by_pos[k] = -1;
      dist_by_pos[k] = 1e30f;
      int ridx=cand[(size_t)qi*RERANK+k]; if(ridx<0) continue;
      int idx=reorder_to_original[(size_t)ridx]; if(idx<0) continue;
      orig_by_pos[k] = idx;
      read_order[nvalid++] = {idx, k};
    }
    std::sort(read_order, read_order+nvalid,
              [](const std::pair<int,int>& a,const std::pair<int,int>& b){
                return a.first < b.first || (a.first == b.first && a.second < b.second);
              });
    for(int r=0;r<nvalid;r++){
#if defined(__GNUC__)
      constexpr int PREFETCH_DIST = 8;
      if(r+PREFETCH_DIST<nvalid){
        const float* pv=base.data()+(size_t)read_order[r+PREFETCH_DIST].first*DIM;
        __builtin_prefetch(pv, 0, 1);
      }
#endif
      int idx=read_order[r].first;
      int pos=read_order[r].second;
      const float* v=base.data()+(size_t)idx*DIM;
      float d=0.f;
      for(int j=0;j<DIM;j++){float x=qq[j]-v[j]; d+=x*x;}
      dist_by_pos[pos]=d;
    }
    for(int k=0;k<RERANK;k++){
      int idx=orig_by_pos[k]; if(idx<0) continue;
      float d=dist_by_pos[k];
      int worst=0; float wd=topd[0];
      for(int t=1;t<100;t++) if(topd[t]>wd){wd=topd[t]; worst=t;}
      if(d<wd){topd[worst]=d; topi[worst]=idx;}
    }
    for(int a=0;a<100;a++) for(int b=a+1;b<100;b++) if(topd[b]<topd[a]){float td=topd[a]; topd[a]=topd[b]; topd[b]=td; int ti=topi[a]; topi[a]=topi[b]; topi[b]=ti;}
    for(int t=0;t<100;t++){final_idx[(size_t)qi*100+t]=topi[t]; final_dist[(size_t)qi*100+t]=topd[t]; checksum += topd[t] * (t+1);}
  }
  auto t1=std::chrono::high_resolution_clock::now();
  long long hit1=0, hit10=0, hit100=0;
  for(int qi=0; qi<bs; ++qi){
    if(final_idx[(size_t)qi*100] == gt[(size_t)qi*100]) hit1++;
    for(int g=0; g<10; ++g){
      int id=gt[(size_t)qi*100+g];
      for(int r=0;r<10;r++) if(final_idx[(size_t)qi*100+r]==id){hit10++; break;}
    }
    for(int g=0; g<100; ++g){
      int id=gt[(size_t)qi*100+g];
      for(int r=0;r<100;r++) if(final_idx[(size_t)qi*100+r]==id){hit100++; break;}
    }
  }
  g_rerank_checksum_sink = checksum;
  RecallResult rr;
  rr.ms = std::chrono::duration<double,std::milli>(t1-t0).count();
  rr.checksum = checksum;
  rr.recall1 = (double)hit1 / (double)bs;
  rr.recall10 = (double)hit10 / (double)(bs*10);
  rr.recall100 = (double)hit100 / (double)(bs*100);
  if(out) *out = rr;
  return rr.ms;
}

static double read_mem_available_gib(){
  std::ifstream f("/proc/meminfo");
  std::string key, unit; long long value=0;
  while(f>>key>>value>>unit){
    if(key=="MemAvailable:") return (double)value/1024.0/1024.0;
  }
  return -1.0;
}


int main(int argc,char**argv){
  if(argc<4){std::cerr<<"usage: "<<argv[0]<<" sift100m|sift1b out_csv full|update [repeats]\n";return 2;}
  std::string ds=argv[1], out_csv=argv[2], mode=argv[3]; int repeats=argc>=5?atoi(argv[4]):1;
  const bool update_mode = (mode=="update");
  int nlist=0; std::string root, base_path, query_path, gt_path, cent_path, assign_path, cb_path, codes_path;
  std::vector<std::pair<int,int>> configs;
  int bss_full[]={8,2048}; int nps_full[]={160,192,224};
  if(ds=="sift10m"){
    gN=10000000LL; nlist=4096; root="/workspace/results/sift10m";
    base_path="/workspace/sift1b/base_1b.bin"; query_path="/workspace/sift1b/query.bin"; gt_path="/dev/shm/sift1b/groundtruth_10m.bin";
    cent_path=root+"/centroids/centroids_10m_nlist4096.bin"; assign_path=root+"/assign_10m_nlist4096.bin";
    cb_path=root+"/pq/codebook_resid_10m_nlist4096.bin"; codes_path=root+"/pq/pq_codes_resid_10m_nlist4096.bin";
  } else if(ds=="sift100m"){
    gN=100000000LL; nlist=32768; root="/workspace/results/sift100m";
    base_path="/workspace/sift1b/base_1b.bin"; query_path="/workspace/sift1b/query.bin"; gt_path="/workspace/data/sift100m/groundtruth.bin";
    cent_path=root+"/centroids/centroids_100m_nlist32768.bin"; assign_path=root+"/assign_100m_nlist32768.bin";
    cb_path=root+"/pq/codebook_resid_100m_nlist32768.bin"; codes_path=root+"/pq/pq_codes_resid_100m_nlist32768.bin";
  } else if(ds=="sift1b"){
    gN=1000000000LL; nlist=524288; root="/workspace/results/sift1b";
    base_path="/workspace/sift1b/base_1b.bin"; query_path="/workspace/sift1b/query.bin"; gt_path="/workspace/sift1b/gt.bin";
    cent_path="/workspace/results/sift1b/centroids/centroids_1b_nlist524288_train1b_iter10_8gpu.bin"; assign_path="/workspace/results/sift1b/assign_1b_nlist524288_train1b_iter10_8gpu.bin";
    cb_path="/workspace/results/sift1b/pq/codebook_resid_M16_1b_nlist524288_train1b_iter10_8gpu.bin";
    codes_path="/workspace/results/sift1b/pq/pq_codes_resid_M16_1b_nlist524288_train1b_iter10_8gpu.bin";
  } else {std::cerr<<"unsupported dataset "<<ds<<"\n"; return 2;}
  if(mode=="full") for(int bs: bss_full) for(int np: nps_full) configs.push_back({bs,np});
  else if(mode=="update") configs={{8,160},{2048,160}};
  else {std::cerr<<"bad mode\n"; return 2;}

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFuncSetAttribute(group_merge_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,GROUP_CHUNKS*RERANK*(int)(sizeof(float)+sizeof(int))));
  std::cerr<<"mem_available_gib_before_load="<<read_mem_available_gib()<<"\n";
  auto q_all=read_query_u8bin_prefix(query_path,10000);
  auto cent=read_centroids(cent_path,nlist);
  auto assign=read_assign(assign_path);
  auto counts=counts_from_assign(assign,nlist); auto offsets=offsets_from_counts(counts);
  auto reorder_to_original=build_reorder_to_original_from_assign(assign, offsets);
  assign.clear(); assign.shrink_to_fit();
  auto cb=read_codebook(cb_path);
  auto codes=read_codes(codes_path);
  auto base_u8=read_base_u8bin_prefix_memory(base_path, gN);
  auto gt100=read_groundtruth100_anyk(gt_path,10000);
  std::unordered_set<int> delete_protect_gt;
  if(update_mode){
    delete_protect_gt.reserve(gt100.size() * 2);
    for(int id: gt100) if(id>=0) delete_protect_gt.insert(id);
    std::cerr<<"[DELETE] GT-safe protect ids="<<delete_protect_gt.size()<<"\n";
  }
  std::cerr<<"loaded "<<ds<<" full-e2e n="<<gN<<" nlist="<<nlist<<" configs="<<configs.size()<<" repeats="<<repeats<<" mem_available_gib_after_load="<<read_mem_available_gib()<<"\n";

  float *dc=nullptr,*dcb=nullptr; uint8_t* dcodes=nullptr; long long* doff=nullptr; int* dcounts=nullptr;
  CUDA_CHECK(cudaMalloc(&dc,cent.size()*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dcb,cb.size()*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dcodes,codes.size()));
  CUDA_CHECK(cudaMalloc(&doff,offsets.size()*sizeof(long long))); CUDA_CHECK(cudaMalloc(&dcounts,counts.size()*sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dc,cent.data(),cent.size()*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dcb,cb.data(),cb.size()*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dcodes,codes.data(),codes.size(),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(doff,offsets.data(),offsets.size()*sizeof(long long),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dcounts,counts.data(),counts.size()*sizeof(int),cudaMemcpyHostToDevice));
  CoarseHandle h; coarse_handle_init(&h, dc, nlist, DIM);
  std::vector<uint8_t> delta_u8;
  std::vector<DeltaPqGpuSegment> delta_pq_segments;
  if(update_mode){
    const int max_delta_n=50000000;
    delta_u8=make_random_delta_u8(max_delta_n, 20260531ULL);
    for(int dn: {10000000,50000000}){
      auto seg=build_delta_pq_mutable_publish(dn,nlist,(int)gN);
      auto t0=std::chrono::high_resolution_clock::now();
      seg.upload();
      auto t1=std::chrono::high_resolution_clock::now();
      std::cerr<<"[DELTA-PQ-GPU] resident delta_n="<<dn
               <<" pq_codes_mb="<<(double)seg.codes.size()/1000.0/1000.0
               <<" upload_ms="<<std::chrono::duration<double,std::milli>(t1-t0).count()<<"\n";
      delta_pq_segments.push_back(std::move(seg));
    }
  }
  std::ofstream out(out_csv,std::ios::app);
  if(out.tellp()==0){
    if(update_mode) out<<"system,scale,nlist,nprobe,batch_size,insert_n,delete_n,delete_ratio,repeat,nchunks,ngroups,opt_coarse_ms,opt_coarse_h2d_ms,opt_coarse_d2h_ms,opt_query_h2d_ms,opt_gemm_ms,opt_topk_ms,desc_ms,chunk_ms,group_merge_ms,final_merge_ms,candidate_d2h_ms,cpu_rerank_ms,fine_search_ms,total_ms,p50_ms,p99_ms,qps,recall1,recall10,recall100,checksum,mem_free_mib,mem_total_mib,mem_avail_gib\n";
    else out<<"system,scale,nlist,nprobe,batch_size,repeat,nchunks,ngroups,opt_coarse_ms,opt_coarse_h2d_ms,opt_coarse_d2h_ms,opt_query_h2d_ms,opt_gemm_ms,opt_topk_ms,desc_ms,chunk_ms,group_merge_ms,final_merge_ms,candidate_d2h_ms,cpu_rerank_ms,fine_search_ms,total_ms,p50_ms,p99_ms,qps,recall1,recall10,recall100,checksum,mem_free_mib,mem_total_mib,mem_avail_gib\n";
  }
  cudaEvent_t evs,eve; CUDA_CHECK(cudaEventCreate(&evs)); CUDA_CHECK(cudaEventCreate(&eve));
  
  const int nq_eval = std::min<int>(10000, (int)(q_all.size()/DIM));

  int *ws_hcids=nullptr,*ws_pair_counts=nullptr,*ws_pair_offsets=nullptr;
  int *ws_q_counts=nullptr,*ws_q_start=nullptr,*ws_next_counts=nullptr,*ws_next_start=nullptr;
  int *ws_chunki=nullptr,*ws_merge_a_i=nullptr,*ws_merge_b_i=nullptr,*ws_outi=nullptr,*ws_delta_outi=nullptr;
  float *ws_dq=nullptr,*ws_chunkd=nullptr,*ws_merge_a_d=nullptr,*ws_merge_b_d=nullptr,*ws_outd=nullptr,*ws_delta_outd=nullptr;
  ChunkDesc* ws_ddesc=nullptr; GroupDesc* ws_gdesc=nullptr;
  void* ws_scan_tmp=nullptr; size_t ws_scan_tmp_bytes=0;
  size_t cap_hcids=0,cap_pair_counts=0,cap_pair_offsets=0,cap_q_counts=0,cap_q_start=0,cap_next_counts=0,cap_next_start=0;
  size_t cap_chunki=0,cap_merge_a_i=0,cap_merge_b_i=0,cap_outi=0,cap_delta_outi=0,cap_dq=0,cap_chunkd=0,cap_merge_a_d=0,cap_merge_b_d=0,cap_outd=0,cap_delta_outd=0,cap_ddesc=0,cap_gdesc=0;
  auto ensure_int=[&](int** p,size_t& cap,size_t n){ if(n<1)n=1; if(n>cap){ if(*p) CUDA_CHECK(cudaFree(*p)); CUDA_CHECK(cudaMalloc(p,n*sizeof(int))); cap=n; } };
  auto ensure_float=[&](float** p,size_t& cap,size_t n){ if(n<1)n=1; if(n>cap){ if(*p) CUDA_CHECK(cudaFree(*p)); CUDA_CHECK(cudaMalloc(p,n*sizeof(float))); cap=n; } };
  auto ensure_chunk_desc=[&](ChunkDesc** p,size_t& cap,size_t n){ if(n<1)n=1; if(n>cap){ if(*p) CUDA_CHECK(cudaFree(*p)); CUDA_CHECK(cudaMalloc(p,n*sizeof(ChunkDesc))); cap=n; } };
  auto ensure_group_desc=[&](GroupDesc** p,size_t& cap,size_t n){ if(n<1)n=1; if(n>cap){ if(*p) CUDA_CHECK(cudaFree(*p)); CUDA_CHECK(cudaMalloc(p,n*sizeof(GroupDesc))); cap=n; } };
  for(auto cfg: configs){
    int bs=cfg.first, nprobe=cfg.second;
    std::vector<std::pair<int,int>> run_pairs;
    if(update_mode) run_pairs={{0,0},{0,50000000},{50000000,0},{50000000,50000000},{10000000,10000000},{10000000,50000000},{50000000,10000000}};
    else run_pairs={{0,0}};
    for(auto update_pair: run_pairs){
    int active_delta_n=update_pair.first;
    int delete_n=update_pair.second;
    double delete_ratio=(gN>0)?((double)delete_n/(double)gN):0.0;
    const DeltaPqGpuSegment* active_delta_seg=update_mode ? find_delta_pq_segment(delta_pq_segments, active_delta_n) : nullptr;
    const int cand_k = update_mode ? COMBINED_CAND : RERANK;
    for(int rep=0; rep<repeats; ++rep){
      double sum_coarse=0, sum_coarse_h2d=0, sum_coarse_d2h=0;
      double sum_query_h2d=0, sum_gemm=0, sum_topk=0;
      double sum_desc=0, sum_chunk=0, sum_group=0, sum_final=0, sum_d2h=0, sum_rerank=0;
      double sum_fine=0, sum_total=0, checksum=0;
      double acc_r1=0, acc_r10=0, acc_r100=0;
      long long processed=0, total_chunks=0, total_groups=0;
      size_t mem_free=0, mem_total=0;
      std::vector<double> batch_latencies;
      struct PendingRerankJob { std::vector<float> q; std::vector<int> gt; std::vector<int> cand; int bs=0; int cand_k=RERANK; double delete_ratio=0.0; bool update=false; };
      std::mutex rerank_mu;
      std::condition_variable rerank_cv;
      PendingRerankJob rerank_job;
      RecallResult rerank_result{};
      bool rerank_has_job=false;
      bool rerank_done=false;
      bool rerank_stop=false;
      bool has_pending=false;
      bool pending_uses_worker=false;
      int pending_bs=0;
      double pending_start_ms=0.0;
      std::future<RecallResult> pending_future;
      auto rerank_worker_fn = [&](){
        while(true){
          PendingRerankJob job;
          {
            std::unique_lock<std::mutex> lk(rerank_mu);
            rerank_cv.wait(lk,[&](){ return rerank_has_job || rerank_stop; });
            if(rerank_stop && !rerank_has_job) return;
            job=std::move(rerank_job);
            rerank_has_job=false;
          }
          RecallResult rr{};
          if(job.update) rr=cpu_rerank_mem_recall_ms_update(base_u8,delta_u8,job.q,job.cand,job.cand_k,reorder_to_original,job.gt,&delete_protect_gt,job.bs,job.delete_ratio);
          else rr=cpu_rerank_mem_recall_ms(base_u8,job.q,job.cand,reorder_to_original,job.gt,job.bs);
          {
            std::lock_guard<std::mutex> lk(rerank_mu);
            rerank_result=rr;
            rerank_done=true;
          }
          rerank_cv.notify_one();
        }
      };
      std::thread rerank_worker;
      if(bs<512) rerank_worker=std::thread(rerank_worker_fn);
      auto finish_pending = [&](){
        if(!has_pending) return;
        RecallResult rr{};
        if(pending_uses_worker){
        {
          std::unique_lock<std::mutex> lk(rerank_mu);
          rerank_cv.wait(lk,[&](){ return rerank_done; });
          rr=rerank_result;
          rerank_done=false;
        }
        } else {
          rr=pending_future.get();
        }
        sum_rerank += rr.ms;
        sum_fine += rr.ms;
        checksum += rr.checksum;
        acc_r1 += rr.recall1 * pending_bs;
        acc_r10 += rr.recall10 * pending_bs;
        acc_r100 += rr.recall100 * pending_bs;
        processed += pending_bs;
        if(pending_start_ms>0.0) batch_latencies.push_back(wall_now_ms()-pending_start_ms);
        has_pending=false;
        pending_uses_worker=false;
        pending_start_ms=0.0;
      };
      auto submit_pending = [&](std::vector<float>&& q, std::vector<int>&& gt, std::vector<int>&& cand, int run_bs, double batch_start_ms){
        finish_pending();
        pending_bs=run_bs;
        pending_start_ms=batch_start_ms;
        if(bs<512){
        {
          std::lock_guard<std::mutex> lk(rerank_mu);
          rerank_job.q=std::move(q);
          rerank_job.gt=std::move(gt);
          rerank_job.cand=std::move(cand);
          rerank_job.bs=run_bs;
          rerank_job.cand_k=cand_k;
          rerank_job.delete_ratio=delete_ratio;
          rerank_job.update=update_mode;
          rerank_done=false;
          rerank_has_job=true;
          has_pending=true;
          pending_uses_worker=true;
        }
        rerank_cv.notify_one();
        } else {
          pending_future=std::async(std::launch::async, [&, q=std::move(q), gt=std::move(gt), cand=std::move(cand), run_bs]() mutable {
            RecallResult rr{};
          if(update_mode) rr=cpu_rerank_mem_recall_ms_update(base_u8,delta_u8,q,cand,cand_k,reorder_to_original,gt,&delete_protect_gt,run_bs,delete_ratio);
          else rr=cpu_rerank_mem_recall_ms(base_u8,q,cand,reorder_to_original,gt,run_bs);
            return rr;
          });
          has_pending=true;
          pending_uses_worker=false;
        }
      };
      auto stop_rerank_worker = [&](){
        finish_pending();
        {
          std::lock_guard<std::mutex> lk(rerank_mu);
          rerank_stop=true;
        }
        rerank_cv.notify_one();
        if(rerank_worker.joinable()) rerank_worker.join();
      };
      auto wall0=std::chrono::high_resolution_clock::now();
      for(int qoff=0; qoff<nq_eval; qoff+=bs){
        double batch_start_ms=wall_now_ms();
        int run_bs=std::min(bs, nq_eval-qoff);
        std::vector<float> q(q_all.begin()+(size_t)qoff*DIM, q_all.begin()+(size_t)(qoff+run_bs)*DIM);
        std::vector<int> gt(gt100.begin()+(size_t)qoff*100, gt100.begin()+(size_t)(qoff+run_bs)*100);
        std::vector<int> hcids((size_t)run_bs*nprobe);
        double coarse_ms=0, coarse_h2d=0, coarse_d2h=0;
        setenv("IVFT_COARSE_TOPK_IMPL","auto",1);
        bool coarse_tiled_ok = (nprobe==1||nprobe==2||nprobe==4||nprobe==8||nprobe==16||nprobe==32||nprobe==64||nprobe==128||nprobe==256||nprobe==512);
        if(run_bs>=2048 && coarse_tiled_ok){ setenv("IVFT_COARSE_TILED","1",1); }
        else { setenv("IVFT_COARSE_TILED","0",1); }
        coarse_search(&h,q.data(),run_bs,nprobe,0,hcids.data(),&coarse_ms,&coarse_h2d,&coarse_d2h);
        const CoarseTimingBreakdown* bd=coarse_get_last_timing();
        sum_coarse+=coarse_ms; sum_coarse_h2d+=coarse_h2d; sum_coarse_d2h+=coarse_d2h; sum_query_h2d+=(bd?bd->query_h2d_ms:0); sum_gemm+=(bd?bd->gemm_ms:0); sum_topk+=(bd?bd->topk_ms:0);
        double batch_fine_total=0;
        const int fine_query_tile=512;
        std::vector<int> hout_batch((size_t)run_bs*cand_k, -1);
        for(int fine_off=0; fine_off<run_bs; fine_off+=fine_query_tile){
          int fine_bs=std::min(fine_query_tile, run_bs-fine_off);
          std::vector<float> qf(q.begin()+(size_t)fine_off*DIM, q.begin()+(size_t)(fine_off+fine_bs)*DIM);
          std::vector<int> gtf(gt.begin()+(size_t)fine_off*100, gt.begin()+(size_t)(fine_off+fine_bs)*100);
          std::vector<int> hcids_f((size_t)fine_bs*nprobe);
          for(int qi=0; qi<fine_bs; ++qi) for(int p=0; p<nprobe; ++p) hcids_f[(size_t)qi*nprobe+p]=hcids[(size_t)(fine_off+qi)*nprobe+p];
          auto ht0=std::chrono::high_resolution_clock::now();
          const int total_pairs=fine_bs*nprobe;
          ensure_int(&ws_hcids,cap_hcids,(size_t)total_pairs);
          ensure_int(&ws_pair_counts,cap_pair_counts,(size_t)total_pairs);
          ensure_int(&ws_pair_offsets,cap_pair_offsets,(size_t)total_pairs);
          ensure_int(&ws_q_counts,cap_q_counts,(size_t)fine_bs);
          ensure_int(&ws_q_start,cap_q_start,(size_t)fine_bs);
          ensure_int(&ws_next_counts,cap_next_counts,(size_t)fine_bs);
          ensure_int(&ws_next_start,cap_next_start,(size_t)fine_bs);
          ensure_float(&ws_dq,cap_dq,(size_t)fine_bs*DIM);
          ensure_float(&ws_outd,cap_outd,(size_t)fine_bs*RERANK);
          ensure_int(&ws_outi,cap_outi,(size_t)fine_bs*RERANK);
          CUDA_CHECK(cudaMemcpy(ws_hcids,hcids_f.data(),(size_t)total_pairs*sizeof(int),cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(ws_dq,qf.data(),(size_t)fine_bs*DIM*sizeof(float),cudaMemcpyHostToDevice));
          int threads=256, blocks=(total_pairs+threads-1)/threads;
          pair_chunk_count_kernel<<<blocks,threads>>>(ws_hcids,dcounts,total_pairs,ws_pair_counts);
          CUDA_CHECK(cudaGetLastError());
          device_exclusive_scan_int_ws(ws_pair_counts,ws_pair_offsets,total_pairs,&ws_scan_tmp,&ws_scan_tmp_bytes);
          int last_count=0,last_off=0;
          CUDA_CHECK(cudaMemcpy(&last_count,ws_pair_counts+total_pairs-1,sizeof(int),cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(&last_off,ws_pair_offsets+total_pairs-1,sizeof(int),cudaMemcpyDeviceToHost));
          int nchunks=last_off+last_count;
          if(nchunks<=0 || nchunks>30000000){std::cerr<<"bad nchunks from gpu desc build: "<<nchunks<<"\n"; return 3;}
          ensure_chunk_desc(&ws_ddesc,cap_ddesc,(size_t)nchunks);
          emit_chunk_desc_kernel<<<blocks,threads>>>(ws_hcids,dcounts,ws_pair_offsets,total_pairs,nprobe,ws_ddesc);
          CUDA_CHECK(cudaGetLastError());
          query_chunk_count_kernel<<<fine_bs,256>>>(ws_pair_counts,fine_bs,nprobe,ws_q_counts);
          CUDA_CHECK(cudaGetLastError());
          device_exclusive_scan_int_ws(ws_q_counts,ws_q_start,fine_bs,&ws_scan_tmp,&ws_scan_tmp_bytes);
          CUDA_CHECK(cudaDeviceSynchronize());
          auto ht1=std::chrono::high_resolution_clock::now(); double descms=std::chrono::duration<double,std::milli>(ht1-ht0).count();
          float chunkms=0,groupms=0,finalms=0,d2hms=0;
          int* first_count=ws_next_counts;
          int* first_start=ws_next_start;
          group_count_kernel<<<(fine_bs+255)/256,256>>>(ws_q_counts,fine_bs,first_count);
          CUDA_CHECK(cudaGetLastError());
          device_exclusive_scan_int_ws(first_count,first_start,fine_bs,&ws_scan_tmp,&ws_scan_tmp_bytes);
          int fg_last=0,fs_last=0;
          CUDA_CHECK(cudaMemcpy(&fg_last,first_count+fine_bs-1,sizeof(int),cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(&fs_last,first_start+fine_bs-1,sizeof(int),cudaMemcpyDeviceToHost));
          int first_groups=fs_last+fg_last;
          ensure_group_desc(&ws_gdesc,cap_gdesc,(size_t)first_groups);
          emit_group_desc_kernel<<<fine_bs,128>>>(ws_q_start,ws_q_counts,first_start,fine_bs,ws_gdesc);
          CUDA_CHECK(cudaGetLastError());
          ensure_float(&ws_merge_a_d,cap_merge_a_d,(size_t)first_groups*RERANK);
          ensure_int(&ws_merge_a_i,cap_merge_a_i,(size_t)first_groups*RERANK);

          CUDA_CHECK(cudaEventRecord(evs));
          fused_first_group_merge_kernel<<<first_groups,1024,GROUP_CHUNKS*LOCAL_K*(sizeof(float)+sizeof(int))>>>(ws_dq,dc,dcb,dcodes,doff,ws_ddesc,ws_gdesc,first_groups,ws_merge_a_d,ws_merge_a_i);
          CUDA_CHECK(cudaEventRecord(eve)); CUDA_CHECK(cudaEventSynchronize(eve)); CUDA_CHECK(cudaEventElapsedTime(&chunkms,evs,eve)); CUDA_CHECK(cudaGetLastError());
          float* curd=ws_merge_a_d; int* curi=ws_merge_a_i; int* cur_start=first_start; int* cur_count=first_count; size_t cur_groups=first_groups; int cur_stride=RERANK; int initial_ngroups=first_groups; bool first_group_round=false;
          while(true){
            int* out_count=(cur_count==ws_q_counts)?ws_next_counts:ws_q_counts;
            int* out_start=(cur_start==ws_q_start)?ws_next_start:ws_q_start;
            group_count_kernel<<<(fine_bs+255)/256,256>>>(cur_count,fine_bs,out_count);
            CUDA_CHECK(cudaGetLastError());
            device_exclusive_scan_int_ws(out_count,out_start,fine_bs,&ws_scan_tmp,&ws_scan_tmp_bytes);
            int gc_last=0,gs_last=0;
            CUDA_CHECK(cudaMemcpy(&gc_last,out_count+fine_bs-1,sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&gs_last,out_start+fine_bs-1,sizeof(int),cudaMemcpyDeviceToHost));
            int round_groups=gs_last+gc_last;
            if(first_group_round){initial_ngroups=round_groups; first_group_round=false;}
            if(cur_stride==RERANK && (size_t)round_groups==cur_groups) break;
            ensure_group_desc(&ws_gdesc,cap_gdesc,(size_t)round_groups);
            emit_group_desc_kernel<<<fine_bs,128>>>(cur_start,cur_count,out_start,fine_bs,ws_gdesc);
            CUDA_CHECK(cudaGetLastError());
            float* nextd=nullptr; int* nexti=nullptr;
            if(curd==ws_merge_a_d){ensure_float(&ws_merge_b_d,cap_merge_b_d,(size_t)round_groups*RERANK); ensure_int(&ws_merge_b_i,cap_merge_b_i,(size_t)round_groups*RERANK); nextd=ws_merge_b_d; nexti=ws_merge_b_i;}
            else {ensure_float(&ws_merge_a_d,cap_merge_a_d,(size_t)round_groups*RERANK); ensure_int(&ws_merge_a_i,cap_merge_a_i,(size_t)round_groups*RERANK); nextd=ws_merge_a_d; nexti=ws_merge_a_i;}
            float ms=0; CUDA_CHECK(cudaEventRecord(evs));
            if(cur_stride==LOCAL_K) group_merge_local_kernel<<<round_groups,1024,GROUP_CHUNKS*LOCAL_K*(sizeof(float)+sizeof(int))>>>(ws_gdesc,round_groups,curd,curi,nextd,nexti);
            else group_merge_kernel<<<round_groups,1024,GROUP_CHUNKS*RERANK*(sizeof(float)+sizeof(int))>>>(ws_gdesc,round_groups,curd,curi,nextd,nexti);
            CUDA_CHECK(cudaEventRecord(eve)); CUDA_CHECK(cudaEventSynchronize(eve)); CUDA_CHECK(cudaEventElapsedTime(&ms,evs,eve)); CUDA_CHECK(cudaGetLastError());
            groupms+=ms; cur_stride=RERANK; curd=nextd; curi=nexti; cur_groups=round_groups; cur_start=out_start; cur_count=out_count;
          }
          CUDA_CHECK(cudaEventRecord(evs)); copy_final_groups_kernel<<<fine_bs,256>>>(cur_start,curd,curi,ws_outd,ws_outi,fine_bs); CUDA_CHECK(cudaEventRecord(eve)); CUDA_CHECK(cudaEventSynchronize(eve)); CUDA_CHECK(cudaEventElapsedTime(&finalms,evs,eve)); CUDA_CHECK(cudaGetLastError());
          std::vector<int> hmain((size_t)fine_bs*RERANK);
          CUDA_CHECK(cudaEventRecord(evs)); CUDA_CHECK(cudaMemcpy(hmain.data(),ws_outi,(size_t)fine_bs*RERANK*sizeof(int),cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaEventRecord(eve)); CUDA_CHECK(cudaEventSynchronize(eve)); CUDA_CHECK(cudaEventElapsedTime(&d2hms,evs,eve));
          for(int qi=0; qi<fine_bs; ++qi){
            int* dst=hout_batch.data()+(size_t)(fine_off+qi)*cand_k;
            std::copy(hmain.data()+(size_t)qi*RERANK, hmain.data()+(size_t)(qi+1)*RERANK, dst);
          }
          if(update_mode && active_delta_seg && active_delta_n>0){
            auto dht0=std::chrono::high_resolution_clock::now();
            pair_chunk_count_kernel<<<blocks,threads>>>(ws_hcids,active_delta_seg->d_counts,total_pairs,ws_pair_counts);
            CUDA_CHECK(cudaGetLastError());
            device_exclusive_scan_int_ws(ws_pair_counts,ws_pair_offsets,total_pairs,&ws_scan_tmp,&ws_scan_tmp_bytes);
            int delta_last_count=0,delta_last_off=0;
            CUDA_CHECK(cudaMemcpy(&delta_last_count,ws_pair_counts+total_pairs-1,sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&delta_last_off,ws_pair_offsets+total_pairs-1,sizeof(int),cudaMemcpyDeviceToHost));
            int delta_nchunks=delta_last_off+delta_last_count;
            if(delta_nchunks<0 || delta_nchunks>30000000){std::cerr<<"bad delta nchunks from gpu desc build: "<<delta_nchunks<<"\n"; return 3;}
            if(delta_nchunks>0){
              ensure_chunk_desc(&ws_ddesc,cap_ddesc,(size_t)delta_nchunks);
              emit_chunk_desc_kernel<<<blocks,threads>>>(ws_hcids,active_delta_seg->d_counts,ws_pair_offsets,total_pairs,nprobe,ws_ddesc);
              CUDA_CHECK(cudaGetLastError());
              query_chunk_count_kernel<<<fine_bs,256>>>(ws_pair_counts,fine_bs,nprobe,ws_q_counts);
              CUDA_CHECK(cudaGetLastError());
              device_exclusive_scan_int_ws(ws_q_counts,ws_q_start,fine_bs,&ws_scan_tmp,&ws_scan_tmp_bytes);
              CUDA_CHECK(cudaDeviceSynchronize());
              auto dht1=std::chrono::high_resolution_clock::now();
              descms += std::chrono::duration<double,std::milli>(dht1-dht0).count();

              int* delta_first_count=ws_next_counts;
              int* delta_first_start=ws_next_start;
              group_count_kernel<<<(fine_bs+255)/256,256>>>(ws_q_counts,fine_bs,delta_first_count);
              CUDA_CHECK(cudaGetLastError());
              device_exclusive_scan_int_ws(delta_first_count,delta_first_start,fine_bs,&ws_scan_tmp,&ws_scan_tmp_bytes);
              int dfg_last=0,dfs_last=0;
              CUDA_CHECK(cudaMemcpy(&dfg_last,delta_first_count+fine_bs-1,sizeof(int),cudaMemcpyDeviceToHost));
              CUDA_CHECK(cudaMemcpy(&dfs_last,delta_first_start+fine_bs-1,sizeof(int),cudaMemcpyDeviceToHost));
              int delta_first_groups=dfs_last+dfg_last;
              ensure_group_desc(&ws_gdesc,cap_gdesc,(size_t)delta_first_groups);
              emit_group_desc_kernel<<<fine_bs,128>>>(ws_q_start,ws_q_counts,delta_first_start,fine_bs,ws_gdesc);
              CUDA_CHECK(cudaGetLastError());
              ensure_float(&ws_merge_a_d,cap_merge_a_d,(size_t)delta_first_groups*RERANK);
              ensure_int(&ws_merge_a_i,cap_merge_a_i,(size_t)delta_first_groups*RERANK);

              float delta_chunkms=0.f, delta_groupms=0.f, delta_finalms=0.f, delta_d2h=0.f;
              CUDA_CHECK(cudaEventRecord(evs));
              fused_first_group_merge_kernel<<<delta_first_groups,1024,GROUP_CHUNKS*LOCAL_K*(sizeof(float)+sizeof(int))>>>(ws_dq,dc,dcb,active_delta_seg->d_codes,active_delta_seg->d_offsets,ws_ddesc,ws_gdesc,delta_first_groups,ws_merge_a_d,ws_merge_a_i);
              CUDA_CHECK(cudaEventRecord(eve)); CUDA_CHECK(cudaEventSynchronize(eve)); CUDA_CHECK(cudaEventElapsedTime(&delta_chunkms,evs,eve)); CUDA_CHECK(cudaGetLastError());

              float* dcurd=ws_merge_a_d; int* dcuri=ws_merge_a_i; int* dcur_start=delta_first_start; int* dcur_count=delta_first_count; size_t dcur_groups=delta_first_groups; int dcur_stride=RERANK; int delta_initial_groups=delta_first_groups;
              while(true){
                int* dout_count=(dcur_count==ws_q_counts)?ws_next_counts:ws_q_counts;
                int* dout_start=(dcur_start==ws_q_start)?ws_next_start:ws_q_start;
                group_count_kernel<<<(fine_bs+255)/256,256>>>(dcur_count,fine_bs,dout_count);
                CUDA_CHECK(cudaGetLastError());
                device_exclusive_scan_int_ws(dout_count,dout_start,fine_bs,&ws_scan_tmp,&ws_scan_tmp_bytes);
                int dgc_last=0,dgs_last=0;
                CUDA_CHECK(cudaMemcpy(&dgc_last,dout_count+fine_bs-1,sizeof(int),cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&dgs_last,dout_start+fine_bs-1,sizeof(int),cudaMemcpyDeviceToHost));
                int delta_round_groups=dgs_last+dgc_last;
                if(dcur_stride==RERANK && (size_t)delta_round_groups==dcur_groups) break;
                ensure_group_desc(&ws_gdesc,cap_gdesc,(size_t)delta_round_groups);
                emit_group_desc_kernel<<<fine_bs,128>>>(dcur_start,dcur_count,dout_start,fine_bs,ws_gdesc);
                CUDA_CHECK(cudaGetLastError());
                float* dnextd=nullptr; int* dnexti=nullptr;
                if(dcurd==ws_merge_a_d){ensure_float(&ws_merge_b_d,cap_merge_b_d,(size_t)delta_round_groups*RERANK); ensure_int(&ws_merge_b_i,cap_merge_b_i,(size_t)delta_round_groups*RERANK); dnextd=ws_merge_b_d; dnexti=ws_merge_b_i;}
                else {ensure_float(&ws_merge_a_d,cap_merge_a_d,(size_t)delta_round_groups*RERANK); ensure_int(&ws_merge_a_i,cap_merge_a_i,(size_t)delta_round_groups*RERANK); dnextd=ws_merge_a_d; dnexti=ws_merge_a_i;}
                float dms=0.f; CUDA_CHECK(cudaEventRecord(evs));
                if(dcur_stride==LOCAL_K) group_merge_local_kernel<<<delta_round_groups,1024,GROUP_CHUNKS*LOCAL_K*(sizeof(float)+sizeof(int))>>>(ws_gdesc,delta_round_groups,dcurd,dcuri,dnextd,dnexti);
                else group_merge_kernel<<<delta_round_groups,1024,GROUP_CHUNKS*RERANK*(sizeof(float)+sizeof(int))>>>(ws_gdesc,delta_round_groups,dcurd,dcuri,dnextd,dnexti);
                CUDA_CHECK(cudaEventRecord(eve)); CUDA_CHECK(cudaEventSynchronize(eve)); CUDA_CHECK(cudaEventElapsedTime(&dms,evs,eve)); CUDA_CHECK(cudaGetLastError());
                delta_groupms+=dms; dcur_stride=RERANK; dcurd=dnextd; dcuri=dnexti; dcur_groups=delta_round_groups; dcur_start=dout_start; dcur_count=dout_count;
              }
              ensure_int(&ws_delta_outi,cap_delta_outi,(size_t)fine_bs*RERANK);
              ensure_float(&ws_delta_outd,cap_delta_outd,(size_t)fine_bs*RERANK);
              CUDA_CHECK(cudaEventRecord(evs));
              copy_final_groups_kernel<<<fine_bs,256>>>(dcur_start,dcurd,dcuri,ws_delta_outd,ws_delta_outi,fine_bs);
              CUDA_CHECK(cudaEventRecord(eve)); CUDA_CHECK(cudaEventSynchronize(eve)); CUDA_CHECK(cudaEventElapsedTime(&delta_finalms,evs,eve)); CUDA_CHECK(cudaGetLastError());
              map_delta_candidate_ids_kernel<<<((fine_bs*RERANK)+255)/256,256>>>(ws_delta_outi,fine_bs*RERANK,active_delta_seg->d_ids);
              CUDA_CHECK(cudaGetLastError());

              std::vector<int> hdelta((size_t)fine_bs*RERANK);
              CUDA_CHECK(cudaEventRecord(evs));
              CUDA_CHECK(cudaMemcpy(hdelta.data(),ws_delta_outi,(size_t)fine_bs*RERANK*sizeof(int),cudaMemcpyDeviceToHost));
              CUDA_CHECK(cudaEventRecord(eve)); CUDA_CHECK(cudaEventSynchronize(eve)); CUDA_CHECK(cudaEventElapsedTime(&delta_d2h,evs,eve));
              for(int qi=0; qi<fine_bs; ++qi){
                int* dst=hout_batch.data()+(size_t)(fine_off+qi)*cand_k+RERANK;
                std::copy(hdelta.data()+(size_t)qi*RERANK, hdelta.data()+(size_t)qi*RERANK+DELTA_TOPK, dst);
              }
              chunkms += delta_chunkms;
              groupms += delta_groupms;
              finalms += delta_finalms;
              d2hms += delta_d2h;
              total_chunks += delta_nchunks;
              total_groups += delta_initial_groups;
            } else {
              auto dht1=std::chrono::high_resolution_clock::now();
              descms += std::chrono::duration<double,std::milli>(dht1-dht0).count();
            }
          }
          CUDA_CHECK(cudaMemGetInfo(&mem_free,&mem_total));
          double fine=descms+chunkms+groupms+finalms+d2hms; batch_fine_total+=fine;
          sum_desc+=descms; sum_chunk+=chunkms; sum_group+=groupms; sum_final+=finalms; sum_d2h+=d2hms; sum_fine+=fine;
          total_chunks+=nchunks; total_groups+=initial_ngroups;
        }
        submit_pending(std::move(q),std::move(gt),std::move(hout_batch),run_bs,batch_start_ms);
        sum_total+=coarse_ms+batch_fine_total;
      }
      stop_rerank_worker();
      auto wall1=std::chrono::high_resolution_clock::now();
      sum_total=std::chrono::duration<double,std::milli>(wall1-wall0).count();
      double p50_ms=percentile_ms(batch_latencies,50.0);
      double p99_ms=percentile_ms(batch_latencies,99.0);
      double qps=processed/(sum_total/1000.0);
      if(update_mode){
        out<<"pqgpu_mutable_delta_publish_reuse,"<<ds<<","<<nlist<<","<<nprobe<<","<<bs<<","<<active_delta_n<<","<<delete_n<<","<<delete_ratio<<","<<rep<<","<<total_chunks<<","<<total_groups<<","<<sum_coarse<<","<<sum_coarse_h2d<<","<<sum_coarse_d2h<<","<<sum_query_h2d<<","<<sum_gemm<<","<<sum_topk<<","<<sum_desc<<","<<sum_chunk<<","<<sum_group<<","<<sum_final<<","<<sum_d2h<<","<<sum_rerank<<","<<sum_fine<<","<<sum_total<<","<<p50_ms<<","<<p99_ms<<","<<qps<<","<<(acc_r1/processed)<<","<<(acc_r10/processed)<<","<<(acc_r100/processed)<<","<<checksum<<","<<(mem_free/1048576.0)<<","<<(mem_total/1048576.0)<<","<<read_mem_available_gib()<<"\n";
      } else {
        out<<"optcoarse_gpu_local256_global512_sift1b_pipelined_ablation,"<<ds<<","<<nlist<<","<<nprobe<<","<<bs<<","<<rep<<","<<total_chunks<<","<<total_groups<<","<<sum_coarse<<","<<sum_coarse_h2d<<","<<sum_coarse_d2h<<","<<sum_query_h2d<<","<<sum_gemm<<","<<sum_topk<<","<<sum_desc<<","<<sum_chunk<<","<<sum_group<<","<<sum_final<<","<<sum_d2h<<","<<sum_rerank<<","<<sum_fine<<","<<sum_total<<","<<p50_ms<<","<<p99_ms<<","<<qps<<","<<(acc_r1/processed)<<","<<(acc_r10/processed)<<","<<(acc_r100/processed)<<","<<checksum<<","<<(mem_free/1048576.0)<<","<<(mem_total/1048576.0)<<","<<read_mem_available_gib()<<"\n";
      }
      out.flush();
      std::cerr<<"fusedfirst_pipeline cfg bs="<<bs<<" np="<<nprobe<<" insert="<<active_delta_n<<" delete="<<delete_n<<" rep="<<rep<<" processed="<<processed<<" total="<<sum_total<<" qps="<<qps<<" r10="<<(acc_r10/processed)<<"\n";
    }
    }
  }

  for(auto& seg: delta_pq_segments) seg.release();
  coarse_handle_release(&h); cudaFree(dc); cudaFree(dcb); cudaFree(dcodes); cudaFree(doff); cudaFree(dcounts); cudaFree(ws_hcids); cudaFree(ws_pair_counts); cudaFree(ws_pair_offsets); cudaFree(ws_q_counts); cudaFree(ws_q_start); cudaFree(ws_next_counts); cudaFree(ws_next_start); cudaFree(ws_chunki); cudaFree(ws_merge_a_i); cudaFree(ws_merge_b_i); cudaFree(ws_outi); cudaFree(ws_delta_outi); cudaFree(ws_dq); cudaFree(ws_chunkd); cudaFree(ws_merge_a_d); cudaFree(ws_merge_b_d); cudaFree(ws_outd); cudaFree(ws_delta_outd); cudaFree(ws_ddesc); cudaFree(ws_gdesc); if(ws_scan_tmp) cudaFree(ws_scan_tmp); return 0;
}
