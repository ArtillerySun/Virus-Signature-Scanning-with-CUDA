#include "kseq/kseq.h"
#include "common.h"

#include <iostream>

const int P = 97;

__device__ inline bool check(char e, char f) {
    return (e == 'N') | (f == 'N') | (e == f);
}

__global__ void get_phred(
    const int len, 
    const char* d_samples_quals, 
    unsigned char* d_sample_phred_score) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = id; i < len; i += stride) {
        d_sample_phred_score[i] = (unsigned char)d_samples_quals[i] - 33;
    }
}

__global__ void get_hash(
    const int len, 
    const unsigned char* d_samples_phred_score, 
    const int* d_samples_offset, 
    unsigned char* d_samples_hash) {

    int sample_idx = blockIdx.x;
    if (sample_idx >= len) return;

    int start_offset = d_samples_offset[sample_idx];
    int end_offset = d_samples_offset[sample_idx + 1];
    int sample_len = end_offset - start_offset;

    extern __shared__ int sh_sum[];
    int thread_sum = 0;

    sh_sum[threadIdx.x] = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < sample_len; i += blockDim.x) {
        thread_sum += d_samples_phred_score[start_offset + i];
    }
    sh_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = (blockDim.x >> 1); s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_sum[threadIdx.x] += sh_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_samples_hash[sample_idx] = sh_sum[0] % P;
    }
}

struct TmpResult {
    int sample, signature;
    double match_score;
    int hash;
};

__global__ void solve_matcher(
    const int sample_start_idx,
    const int SAMPLES_SIZE,
    const int SIGNATURES_SIZE,
    const unsigned char* __restrict__ d_samples_hash, 
    const unsigned char* __restrict__ d_samples_phred_score, 
    const int* __restrict__ d_samples_offset, 
    const char* __restrict__ d_samples_seqs, 
    const int* __restrict__ d_signatures_offset, 
    const char* __restrict__ d_signatures_seqs, 
    TmpResult* __restrict__ tmpResult) {
    
    int sample_idx = sample_start_idx + blockIdx.x ;
    int signature_idx = blockIdx.y;

    if (sample_idx >= SAMPLES_SIZE) return;
    if (signature_idx >= SIGNATURES_SIZE) return;
    
    int sample_start_offset = d_samples_offset[sample_idx];
    int sample_end_offset = d_samples_offset[sample_idx + 1];
    int sample_len = sample_end_offset - sample_start_offset;

    int signature_start_offset = d_signatures_offset[signature_idx];
    int signature_end_offset = d_signatures_offset[signature_idx + 1];
    int signature_len = signature_end_offset - signature_start_offset;

    int search_space = sample_len - signature_len + 1;

    extern __shared__ __align__(8) char sh_mem[];
    double* sh_max = reinterpret_cast<double*>(sh_mem);
    char* sh_signature_seq = reinterpret_cast<char*>(sh_max + blockDim.x);

    for (int i = threadIdx.x; i < signature_len; i += blockDim.x) {
        sh_signature_seq[i] = d_signatures_seqs[signature_start_offset + i];
    }

    sh_max[threadIdx.x] = -1.0;
    __syncthreads();

    for (int i = threadIdx.x; i < search_space; i += blockDim.x) {
        bool flag = true;

        int j = 0;
        for (; j + 3 < signature_len; j += 4) {

            char s0 = d_samples_seqs[sample_start_offset + i + j];
            char s1 = d_samples_seqs[sample_start_offset + i + j + 1];
            char s2 = d_samples_seqs[sample_start_offset + i + j + 2];
            char s3 = d_samples_seqs[sample_start_offset + i + j + 3];

            char g0 = sh_signature_seq[j];
            char g1 = sh_signature_seq[j + 1];
            char g2 = sh_signature_seq[j + 2];
            char g3 = sh_signature_seq[j + 3];

            if (!check(s0, g0) || !check(s1, g1) || !check(s2, g2) || !check(s3, g3)) {
                flag = false;
                break;
            }
        }

        // for the tail
        for (; flag && j < signature_len; j++) {
            char s = d_samples_seqs[sample_start_offset + i + j];
            char g = sh_signature_seq[j];
            if (!check(s, g)) { flag = false; break; }
        }

        if (!flag) continue;

        double sum = 0;
        j = 0;
        
        for (; j + 3 < signature_len; j += 4) {
            sum += d_samples_phred_score[sample_start_offset + i + j];
            sum += d_samples_phred_score[sample_start_offset + i + j + 1];
            sum += d_samples_phred_score[sample_start_offset + i + j + 2];
            sum += d_samples_phred_score[sample_start_offset + i + j + 3];
        }

        // for the tail
        for (; j < signature_len; j++) {
            sum += d_samples_phred_score[sample_start_offset + i + j];
        }

        sh_max[threadIdx.x] = fmax(sh_max[threadIdx.x], sum/signature_len);
    }
    __syncthreads();

    for (int s = (blockDim.x >> 1); s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_max[threadIdx.x] = fmax(sh_max[threadIdx.x], sh_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (sh_max[0] >= 0) {
            tmpResult[sample_idx * SIGNATURES_SIZE + signature_idx] = {sample_idx, signature_idx, sh_max[0], d_samples_hash[sample_idx]};
        }
    }
    
}

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {
    const int SIGNATURES_SIZE = signatures.size();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int total_signatures_len = 0;
    for (const auto& signature : signatures) {
        total_signatures_len += signature.seq.length();
    }

    int* h_signatures_offset_pinned;
    char* h_signatures_seqs_pinned;
    cudaHostAlloc(&h_signatures_seqs_pinned, total_signatures_len * sizeof(char), cudaHostAllocDefault);
    cudaHostAlloc(&h_signatures_offset_pinned, (SIGNATURES_SIZE + 1) * sizeof(int), cudaHostAllocDefault);

    int max_signature_len = 0;
    int current_offset = 0;

    h_signatures_offset_pinned[0] = 0;
    for (int i = 0; i < SIGNATURES_SIZE; i++) {
        const auto& signature = signatures[i];
        int len = signature.seq.length();
        max_signature_len = std::max(max_signature_len, len);
        memcpy(h_signatures_seqs_pinned + current_offset, signature.seq.c_str(), len);
        
        current_offset += len;
        h_signatures_offset_pinned[i + 1] = current_offset;
    }

    // signatures on device
    char *d_signatures_seqs;
    int *d_signatures_offset;

    // allocate memory for signatures on device
    cudaMalloc(&d_signatures_seqs, total_signatures_len * sizeof(char));
    cudaMalloc(&d_signatures_offset, (SIGNATURES_SIZE + 1) * sizeof(int));

    // copy signatures to device
    cudaMemcpyAsync(d_signatures_seqs, h_signatures_seqs_pinned, total_signatures_len * sizeof(char), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_signatures_offset, h_signatures_offset_pinned, (SIGNATURES_SIZE + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    cudaDeviceSynchronize();
    
    const int BATCH_SIZE = 1024;

    for (int batch_start = 0; batch_start < samples.size(); batch_start += BATCH_SIZE) {

        int batch_end = std::min(batch_start + BATCH_SIZE, (int)samples.size());
        const int SAMPLES_SIZE = batch_end - batch_start;

        auto samples_begin = samples.begin() + batch_start;
        auto samples_end = samples.begin() + batch_end;

        int total_samples_len = 0;
        for (auto sample = samples_begin; sample != samples_end; sample++) {
            total_samples_len += sample->seq.length();
        }

        char *h_samples_seqs_pinned, *h_samples_quals_pinned;
        int *h_samples_offset_pinned;
        TmpResult* h_tmpResult_pinned;

        cudaHostAlloc(&h_samples_seqs_pinned, total_samples_len * sizeof(char), cudaHostAllocDefault);
        cudaHostAlloc(&h_samples_quals_pinned, total_samples_len * sizeof(char), cudaHostAllocDefault);
        cudaHostAlloc(&h_samples_offset_pinned, (SAMPLES_SIZE + 1) * sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc(&h_tmpResult_pinned, SAMPLES_SIZE * SIGNATURES_SIZE * sizeof(TmpResult), cudaHostAllocDefault);

        int current_offset = 0;

        int cnt = 0;
        h_samples_offset_pinned[cnt] = 0;
        for (auto sample = samples_begin; sample != samples_end; sample++) {
            int len = sample->seq.length();
            memcpy(h_samples_seqs_pinned + current_offset, sample->seq.c_str(), len);
            memcpy(h_samples_quals_pinned + current_offset, sample->qual.c_str(), len);
            
            current_offset += len;
            h_samples_offset_pinned[++cnt] = current_offset;
        }

        for (int i = 0; i < SAMPLES_SIZE * SIGNATURES_SIZE; i++) {
            h_tmpResult_pinned[i].match_score = -1.0f;
        }

        // samples on device
        char *d_samples_quals, *d_samples_seqs;
        unsigned char* d_samples_phred_score, *d_samples_hash;
        int *d_samples_offset;

        TmpResult* d_tmpResult;

        // allocate memory for samples on device
        cudaMalloc(&d_samples_seqs, total_samples_len * sizeof(char));
        cudaMalloc(&d_samples_offset, (SAMPLES_SIZE + 1) * sizeof(int));
        cudaMalloc(&d_samples_hash, SAMPLES_SIZE * sizeof(unsigned char));
        cudaMalloc(&d_samples_quals, total_samples_len * sizeof(char));
        cudaMalloc(&d_samples_phred_score, total_samples_len * sizeof(unsigned char));
        cudaMalloc(&d_tmpResult, SAMPLES_SIZE * SIGNATURES_SIZE * sizeof(TmpResult));

        // copy samples to device
        cudaMemcpyAsync(d_samples_seqs, h_samples_seqs_pinned, total_samples_len * sizeof(char), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_samples_offset, h_samples_offset_pinned, (SAMPLES_SIZE + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_samples_quals, h_samples_quals_pinned, total_samples_len * sizeof(char), cudaMemcpyHostToDevice, stream);

        cudaMemcpyAsync(d_tmpResult, h_tmpResult_pinned, SAMPLES_SIZE * SIGNATURES_SIZE * sizeof(TmpResult), cudaMemcpyHostToDevice, stream);

        // calculate phred score with the +33 version
        int blk_size = 256;
        int blk_num = (total_samples_len + blk_size - 1) / blk_size;
        get_phred<<<blk_num, blk_size, 0, stream>>>(total_samples_len, d_samples_quals, d_samples_phred_score);
        
        // calculate hash value for all samples
        blk_num = SAMPLES_SIZE;
        get_hash<<<blk_num, blk_size, blk_size * sizeof(int), stream>>>(SAMPLES_SIZE, d_samples_phred_score, d_samples_offset, d_samples_hash);

        // match the signatures
        int sharedBytes = blk_size * sizeof(double) + max_signature_len * sizeof(char);
        sharedBytes = (sharedBytes + 7) & ~int(7);

        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 16*1024*1024);
        const int B = 128;
        for (int s0 = 0; s0 < SAMPLES_SIZE; s0 += B) {
            int s1 = std::min(s0 + B, SAMPLES_SIZE);

            // calc continous segments
            int batch_off   = h_samples_offset_pinned[s0];
            int batch_end   = h_samples_offset_pinned[s1];
            int batch_bytes = batch_end - batch_off;

            cudaStreamAttrValue apw{};
            apw.accessPolicyWindow.base_ptr  = (void*)(d_samples_seqs + batch_off);
            apw.accessPolicyWindow.num_bytes = std::min(batch_bytes, 16*1024*1024);
            apw.accessPolicyWindow.hitRatio  = 1.0;
            apw.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
            apw.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &apw);

            // grid.x=signatures, grid.y=B
            dim3 grid(s1 - s0, SIGNATURES_SIZE);
            solve_matcher<<<grid, blk_size, sharedBytes, stream>>>(s0, SAMPLES_SIZE, SIGNATURES_SIZE, d_samples_hash, d_samples_phred_score, d_samples_offset, d_samples_seqs, d_signatures_offset, d_signatures_seqs, d_tmpResult);
        
        }
        cudaMemcpyAsync(h_tmpResult_pinned, d_tmpResult, SAMPLES_SIZE * SIGNATURES_SIZE * sizeof(TmpResult), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        for (int i = 0; i < SAMPLES_SIZE; i ++) {
            for (int j = 0; j < SIGNATURES_SIZE; j ++) {
                auto& cur = h_tmpResult_pinned[i * SIGNATURES_SIZE + j];
                if (cur.match_score >= 0) {
                    matches.push_back({samples[batch_start + i].name, signatures[j].name, cur.match_score, cur.hash});
                }
            }
        }
        
        // release all the allocated memory on device
        cudaFree(d_samples_seqs);
        cudaFree(d_samples_offset);
        cudaFree(d_samples_hash);
        cudaFree(d_samples_quals);
        cudaFree(d_samples_phred_score);
        cudaFree(d_tmpResult);

        cudaFreeHost(h_samples_seqs_pinned);
        cudaFreeHost(h_samples_quals_pinned);
        cudaFreeHost(h_samples_offset_pinned);
        cudaFreeHost(h_tmpResult_pinned);
    }

    cudaStreamDestroy(stream);

    cudaFree(d_signatures_seqs);
    cudaFree(d_signatures_offset);

    cudaFreeHost(h_signatures_seqs_pinned);
    cudaFreeHost(h_signatures_offset_pinned);
}
