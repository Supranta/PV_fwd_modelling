module purge

echo 'Loading gcc, python'
module load gcc/9.3.0 intel/18.0-python-3.6.3
echo 'Loading gsl'
module load gsl/2.5
echo 'Loading boost'
module load boost/1.73.0-gcc6
echo 'Loading openmpi'
module load openmpi/3.0.3-ifort-18.0

module load cuda/10.1 tensorflow/2.1
export PYTHONPATH='/home/sarmabor/.local/lib/python3.6/site-packages/jax'
export PYTHONPATH='/home/sarmabor/.local/lib/python3.6/site-packages/jaxlib'
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/softs/cuda/10.1/
