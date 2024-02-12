# Using the slurms

make sure that you have correct python venv

```bash
module load cuda/11.8.0  cmake cudnn/8.9.7.29-cuda nvidia-compilers/23.9 openmpi/4.1.5-cuda
module load python/3.10.4
python -m venv venv
pip cache purge
# Installing mpi4py
CFLAGS=-noswitcherror pip install --no-cache-dir mpi4py
# Installing jax
pip install --no-cache-dir --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
