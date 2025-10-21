#!/bin/bash

#SBATCH --job-name=ass2-sbatch
#SBATCH --time=00:10:00
#SBATCH --output=ass2%j.slurmlog
#SBATCH --error=ass2%j.slurmlog
#SBATCH --gpus=h100-96
#SBATCH --constraint=xgpi
#SBATCH --mail-type=NONE

echo "Running job: $SLURM_JOB_NAME!"
echo "We are running on $(hostname)"
echo "Job started at $(date)"

echo ">> 100k length samp, 1k sig with length 3k, 0.1 ratio"
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./matcher samp_3k100k01.fastq sig_3k01.fasta

echo ">> 100k length samp, 1k sig with length 6k, 0.1 ratio"
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./matcher samp_6k100k01.fastq sig_6k01.fasta

echo ">> 100k length samp, 1k sig with length 9k, 0.1 ratio"
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./matcher samp_9k100k01.fastq sig_9k01.fasta

echo ">> 500k length samp, 1k sig with length 6k, 0.1 ratio"
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./matcher samp_6k500k01.fastq sig_6k01.fasta

echo ">> 1000k length samp, 1k sig with length 6k, 0.1 ratio"
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./matcher samp_6k1000k01.fastq sig_6k01.fasta

echo ">> 1000k length samp (1010), 1k sig with length 6k, 0.1 ratio"
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./matcher samp_6k1000k01_1010.fastq sig_6k01.fasta

echo ">> 1000k length samp (1515), 1k sig with length 6k, 0.1 ratio"
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./matcher samp_6k1000k01_1515.fastq sig_6k01.fasta

echo ">> 1000k length samp, 500 sig with length 6k, 0.1 ratio"
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./matcher samp_6k1000k01__500.fastq sig_6k01_500.fasta

echo ">> 1000k length samp, 750 sig with length 6k, 0.1 ratio"
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./matcher samp_6k1000k01__750.fastq sig_6k01_750.fasta

echo ">> 500k length samp, 1k sig with length 6k, 0.05 ratio"
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./matcher samp_6k500k005.fastq sig_6k005.fasta

echo ">> 500k length samp, 1k sig with length 6k, 0.01 ratio"
srun --ntasks 1 --cpus-per-task 1 --cpu_bind core --mem 20G --gpus h100-96 --constraint xgpi ./matcher samp_6k500k001.fastq sig_6k001.fasta


echo "Job ended at $(date)"
