git clone https://github.com/google/jax
cd jax

#pip3 install numpy scipy six wheel

python build/build.py --enable_cuda
pip3 install dist/*.whl  # installs jaxlib (includes XLA)

pip3 install -e .
cd ..