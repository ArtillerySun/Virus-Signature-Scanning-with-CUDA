# ⚡ CUDA Parallel Signature Matcher

[![CUDA](https://img.shields.io/badge/CUDA-Enabled-success?logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![Language](https://img.shields.io/badge/Language-CUDA%20C++-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Build](https://img.shields.io/badge/Build-Makefile-lightgrey)]()

> GPU-accelerated signature matching using CUDA.  
> Efficient, scalable, and optimized for real-world large dataset matching tasks.

---

## 🧭 Table of Contents
- [Overview](#-overview)
- [Algorithm Pipeline](#-algorithm-pipeline)
- [Parallelization Strategy](#-parallelization-strategy)
- [Optimizations](#-optimizations)
- [Performance Insights](#-performance-insights)
- [Build & Run](#-build--run)
- [Example Dataset](#-example-dataset)
- [Tech Stack](#-tech-stack)
- [Related Resources](#-related-resources)
- [License](#-license)

---

## 🚀 Overview

This project implements a **GPU-parallel signature matching system** designed for high-throughput processing of biological or textual sequence data.  
It performs data conversion, hashing, and sliding-window substring matching using CUDA kernels optimized for concurrency and memory efficiency.

The design emphasizes:
- Asynchronous kernel execution  
- Shared memory caching  
- Efficient data batching for large datasets  
- Wildcard-tolerant matching logic  

---

## 🔬 Algorithm Pipeline

### 1️⃣ Phred Score Conversion (`get_phred`)
Converts ASCII-encoded Phred+33 quality scores into numeric form for later processing.

### 2️⃣ Integrity Hash Computation (`get_hash`)
Computes a modulo-97 checksum per sample using parallel reduction in shared memory to ensure data integrity.

### 3️⃣ Signature Matching (`solve_matcher`)
The main kernel:
- Each **block** matches one sample–signature pair.
- Threads scan distinct windows of the sample in parallel.
- `'N'` is treated as a wildcard.
- A reduction finds the match with the highest confidence score.

---

## ⚙️ Parallelization Strategy

- **Pipeline:** `get_phred → get_hash → solve_matcher`
- **Block Size:** 256 threads (warp-aligned)
- **Grid Configuration:**
  - `get_phred`: `(total_samples_len + blk_size - 1) / blk_size`
  - `get_hash`: one block per sample
  - `solve_matcher`: 2D grid `(num_samples, num_signatures)`
- **Batching:** Handles large datasets without exceeding GPU memory.
- **Pinned Memory:** Host memory allocated via `cudaHostAlloc` for high-speed transfers.
- **Shared Memory:** Used for both caching and reduction operations.

---

## 🧠 Optimizations

### 🔹 Pinned Memory
Replacing pageable memory with pinned memory (`cudaHostAlloc`) enables direct DMA transfers, significantly improving transfer throughput and overlapping data transfer with computation.

### 🔹 Loop Unrolling & Two-Phase Matching
- Manual loop unrolling (×4) improved instruction-level parallelism.  
- Split comparison and scoring phases to minimize warp divergence.  
- Results:
  - Runtime: **22.48 s → 13.04 s**
  - L2 cache hit rate: **22.7% → 33.9%**
  - DRAM traffic: **245 GB → 101 GB**

---

## 📈 Performance Insights

| Factor | Observation |
|--------|--------------|
| **Samples × Signatures** | Runtime grows linearly with task count. |
| **Sample Length** | Dominant cost — longer samples → more comparisons. |
| **Signature Length** | Minor effect due to early mismatch breaks. |
| **‘N’ Wildcards** | Higher ratios slightly increase average runtime. |

---

## 🧪 Build & Run

### Requirements
- NVIDIA GPU with CUDA support  
- CUDA Toolkit ≥ 11.0  
- `make`, `g++`

### Build
```bash
make
````

### Generate Test Data & Run

```bash
./gen_sig <num_signatures> <min_length> <max_length> <n_ratio> > <fasta_file>

./gen_sample <fasta_file> <num_no_virus> <num_with_virus> <min_viruses> <max_viruses> <min_length> <max_length> <min_phred> <max_phred> <n_ratio> > <fastq_file>

# Run the CUDA matcher
./matcher <fastq_file> <fasta_file>
```
---

## 📊 Example Dataset

| Parameter  | Range     | Description          |
| ---------- | --------- | -------------------- |
| Samples    | 1000–2000 | Each 100–2000 KB     |
| Signatures | 500–1000  | Each 3–10 KB         |
| N-ratio    | 0.01–0.10 | Wildcard probability |

---

## 🧰 Tech Stack

* **Language:** CUDA C++
* **Parallelism:** Grid-stride loops, shared memory reduction, warp-level synchronization
* **Optimization:** Pinned memory, loop unrolling, asynchronous kernel pipeline

---

## 🔗 Related Resources

* [Raw Data (Google Sheets)](https://docs.google.com/spreadsheets/d/173EhBG5FrCEOqZYF1mLPmehFsPNLcst-DjjOt2DPYLg/edit?usp=sharing)
* [Batch Script Example](https://github.com/nus-cs3210-students/cs3210-2510-a2-a2-e1297741-e1297745/blob/main/sbatch.sh)