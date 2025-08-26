#!/bin/bash
#SBATCH -p test
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-01:00
#SBATCH -J mergertable
#SBATCH -o mergertable.out
#SBATCH -e mergertable.err
#SBATCH --mem=5G #shouldnt take much memory -- seff <jobid> says it only needs 2.5
#SBATCH --ntasks 32 #number of cores 
#SBATCH --mail-type=ALL

# --- Set up software environment ---
module load python/3.10.9-fasrc01
# module load cuda/11.1.0-fasrc01
# module load cudnn/8.1.0.77_cuda11.2-fasrc01
#module load cudnn/8.2.2.26_cuda11.4-fasrc01
source activate py38
echo "Started"

#python CNNv1.py --epochs 100  --batch_size 2  --n_gpus 2
python build_merger_catalog_TNG_SF.py
#python CNNv1.py -- epochs 5 -- batch_size 256 -- n_gpus 1
#python CNNv1.py -- epochs 100 -- batch_size 32 -- n_gpus 2   #try with 16 per gpu
#python CNNv1.py -- epochs 5 -- batch_size 1024 -- n_gpus 4
# --- Run the code ---
#srun -n 1 --gres=gpu:1 python CNNv1.py 
echo "Done!"