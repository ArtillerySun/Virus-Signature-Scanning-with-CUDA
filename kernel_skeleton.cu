#include "kseq/kseq.h"
#include "common.h"

const int P = 97;

__device__ inline bool check(char e, char f) {
    return e == 'N' || f == 'N' || e == f;
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
    const size_t* d_samples_offset, 
    unsigned char* d_samples_hash) {

    int sample_idx = blockIdx.x;
    if (sample_idx >= len) return;

    size_t start_offset = d_samples_offset[sample_idx];
    size_t end_offset = d_samples_offset[sample_idx + 1];
    size_t sample_len = end_offset - start_offset;

    extern __shared__ int sh_sum[];
    int thread_sum = 0;

    for (size_t i = threadIdx.x; i < sample_len; i += blockDim.x) {
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
    size_t sample, signature;
    float match_score;
    int hash;
};

__global__ void matcher(
    const int SAMPLES_SIZE,
    const int SIGNATURES_SIZE,
    const unsigned char* d_samples_hash, 
    const unsigned char* d_samples_phred_score, 
    const size_t* d_samples_offset, 
    const char* d_samples_seqs, 
    const size_t* d_signatures_offset, 
    const char* d_signatures_seqs, 
    TmpResult* tmpResult) {
    
    size_t sample_idx = blockIdx.x ;
    size_t signature_idx = blockIdx.y;

    if (sample_idx >= SAMPLES_SIZE) return;
    if (signature_idx >= SIGNATURES_SIZE) return;
    
    size_t sample_start_offset = d_samples_offset[sample_idx];
    size_t sample_end_offset = d_samples_offset[sample_idx + 1];
    size_t sample_len = sample_end_offset - sample_start_offset;

    size_t signature_start_offset = d_signatures_offset[signature_idx];
    size_t signature_end_offset = d_signatures_offset[signature_idx + 1];
    size_t signature_len = signature_end_offset - signature_start_offset;

    size_t search_space = sample_len - signature_len + 1;


    extern __shared__ char sh_mem[];
    float* sh_max = (float*)(sh_mem);
    char* sh_signature_seq = (char*)&sh_max[blockDim.x];

    for (size_t i = threadIdx.x; i < signature_len; i += blockDim.x) {
        sh_signature_seq[i] = d_signatures_seqs[signature_start_offset + i];
    }

    sh_max[threadIdx.x] = -1.0;
    __syncthreads();

    for (size_t i = threadIdx.x; i < search_space; i += blockDim.x) {
        float sum = 0;
        bool flag = true;
        for (size_t j = 0; j < signature_len; j++) {
            if (!check(d_samples_seqs[sample_start_offset + i + j], sh_signature_seq[j])) {
                flag = false;
                break;
            }
            sum += d_samples_phred_score[sample_start_offset + i + j];
        }
        if (flag) {
            sh_max[threadIdx.x] = fmaxf(sh_max[threadIdx.x], sum/signature_len);
        }
    }
    __syncthreads();

    for (int s = (blockDim.x >> 1); s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_max[threadIdx.x] = fmaxf(sh_max[threadIdx.x], sh_max[threadIdx.x + s]);
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
    
    const int SAMPLES_SIZE = samples.size();
    const int SIGNATURES_SIZE = signatures.size();
    // samples on host
    std::vector<char> h_samples_seqs;
    std::vector<char> h_samples_quals;
    std::vector<size_t> h_samples_offset;

    // signatures on host
    std::vector<char> h_signatures_seqs;
    std::vector<size_t> h_signatures_offset;

    std::vector<TmpResult> h_tmpResult(SAMPLES_SIZE * SIGNATURES_SIZE);

    for (auto& res : h_tmpResult) {
        res.match_score = -1.0f;
    }

    size_t max_sample_len = 0, max_signature_len = 0;

    size_t cur_offset = 0;
    h_samples_offset.push_back(cur_offset);
    for (auto& sample : samples) {
        size_t cur_len = sample.seq.length();
        max_sample_len = std::max(max_sample_len, cur_len);
        h_samples_seqs.insert(h_samples_seqs.end(), sample.seq.begin(), sample.seq.end());
        h_samples_quals.insert(h_samples_quals.end(), sample.qual.begin(), sample.qual.end());
        cur_offset += cur_len;
        h_samples_offset.push_back(cur_offset);
    }

    cur_offset = 0;
    h_signatures_offset.push_back(cur_offset);
    for (auto& signature : signatures) {
        size_t cur_len = signature.seq.length();
        max_signature_len = std::max(max_signature_len, cur_len);
        h_signatures_seqs.insert(h_signatures_seqs.end(), signature.seq.begin(), signature.seq.end());
        cur_offset += cur_len;
        h_signatures_offset.push_back(cur_offset);
    }

    // samples on device
    char *d_samples_quals, *d_samples_seqs;
    unsigned char* d_samples_phred_score, *d_samples_hash;
    size_t total_samples_len = h_samples_seqs.size();
    size_t *d_samples_offset;

    // signatures on device
    char *d_signatures_seqs;
    size_t total_signatures_len = h_signatures_seqs.size();
    size_t *d_signatures_offset;

    TmpResult* d_tmpResult;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // allocate memory for samples on device
    cudaMalloc(&d_samples_seqs, total_samples_len * sizeof(char));
    cudaMalloc(&d_samples_offset, (SAMPLES_SIZE + 1) * sizeof(size_t));
    cudaMalloc(&d_samples_hash, SAMPLES_SIZE * sizeof(unsigned char));
    cudaMalloc(&d_samples_quals, total_samples_len * sizeof(char));
    cudaMalloc(&d_samples_phred_score, total_samples_len * sizeof(unsigned char));

    // allocate memory for signatures on device
    cudaMalloc(&d_signatures_seqs, total_signatures_len * sizeof(char));
    cudaMalloc(&d_signatures_offset, (SIGNATURES_SIZE + 1) * sizeof(size_t));
    cudaMalloc(&d_tmpResult, SAMPLES_SIZE * SIGNATURES_SIZE * sizeof(TmpResult));

    // copy samples to device
    cudaMemcpyAsync(d_samples_seqs, h_samples_seqs.data(), total_samples_len * sizeof(char), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_samples_offset, h_samples_offset.data(), (SAMPLES_SIZE + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_samples_quals, h_samples_quals.data(), total_samples_len * sizeof(char), cudaMemcpyHostToDevice, stream);

    // copy signatures to device
    cudaMemcpyAsync(d_signatures_seqs, h_signatures_seqs.data(), total_signatures_len * sizeof(char), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_signatures_offset, h_signatures_offset.data(), (SIGNATURES_SIZE + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    
    cudaMemcpyAsync(d_tmpResult, h_tmpResult.data(), SAMPLES_SIZE * SIGNATURES_SIZE * sizeof(TmpResult), cudaMemcpyHostToDevice, stream);

    // calculate phred score with the +33 version
    int blk_size = 256;
    int blk_num = (total_samples_len + blk_size - 1) / blk_size;
    get_phred<<<blk_num, blk_size, 0, stream>>>(total_samples_len, d_samples_quals, d_samples_phred_score);
    
    // calculate hash value for all samples
    blk_num = SAMPLES_SIZE;
    get_hash<<<blk_num, blk_size, blk_size * sizeof(int), stream>>>(SAMPLES_SIZE, d_samples_phred_score, d_samples_offset, d_samples_hash);

    // match the signatures
    dim3 grid(SAMPLES_SIZE, SIGNATURES_SIZE);
    matcher<<<grid, blk_size, max_signature_len * sizeof(char) + blk_size * sizeof(float), stream>>>(SAMPLES_SIZE, SIGNATURES_SIZE, d_samples_hash, d_samples_phred_score, d_samples_offset, d_samples_seqs, d_signatures_offset, d_signatures_seqs, d_tmpResult);
    
    cudaMemcpyAsync(h_tmpResult.data(), d_tmpResult, SAMPLES_SIZE * SIGNATURES_SIZE * sizeof(TmpResult), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < SAMPLES_SIZE; i ++) {
        for (int j = 0; j < SIGNATURES_SIZE; j ++) {
            auto& cur = h_tmpResult[i * SIGNATURES_SIZE + j];
            if (cur.match_score >= 0) {
                matches.push_back({samples[i].name, signatures[j].name, cur.match_score, cur.hash});
            }
        }
    }
    
    cudaStreamDestroy(stream);
    // release all the allocated memory on device
    cudaFree(d_samples_seqs);
    cudaFree(d_samples_offset);
    cudaFree(d_samples_hash);
    cudaFree(d_samples_quals);
    cudaFree(d_samples_phred_score);
    cudaFree(d_signatures_seqs);
    cudaFree(d_signatures_offset);
    cudaFree(d_tmpResult);

}
