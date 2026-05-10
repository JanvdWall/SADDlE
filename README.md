# SADDlE

## Installation
If you are using conda, create an environment and install `pip`.
```bash
conda create -n saddle
conda activate saddle
conda install pip
```
Inside your conda environment, you can install SADDlE using pip.
Change into the folder `SADDlE` and run the following command:

```bash
pip install -e .
```
This will install all the python packages required to run SADDlE.
Notice that this will install the `mpi4py`package.
You can now either configure this `mpi4py` to use a locally installed `MPI` instance or you can use for example `conda` to install `MPI` in your conda-environment by running the following command:

```bash
conda install -c conda-forge mpich
```

If you prefer pip you can run the following command:

```bash
python -m pip install mpich
```
For details about the installation and the different flavor of `MPI` please refer to: https://mpi4py.readthedocs.io/en/stable/install.html

If you want to run the examples in the `examples` folder, you will need to install `jupyterlab` and `matplotlib` as well. You can do this by running the following command:

```bash
pip install jupyterlab matplotlib
```