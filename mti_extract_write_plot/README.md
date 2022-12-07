# mti_extract_write_plot

## Setup and installation
The following guidelines assume that the user runs a conda distribution, i.e.,
Anaconda or Miniconda. If these guidelines are followed, all dependencies for
the `mti_extract_write_plot` code will be installed.

### Create a new `pyechem` conda environment
- It is highly recommended to run the code in a conda environment dedicated to
  the pyEchem library. If the user does not already have that, such a conda
  environment, called `pyechem` and using the latest Python 3 version, can be
  created from:
  ```shell
  conda create -n pyechem python=3
  ```

### Activate `pyechem` conda environment
- When the user has a `pyechem` conda environment, the user should activate the
  pyechem conda environment:
  ```shell
  conda activate pyechem
  ```

### Install dependencies
- Navigate to the main `mti_extract_write_plot` directory. Using conda,
  dependencies from the `conda-forge` channel will be installed, when running:
  ```shell
  conda install -c conda-forge --file requirements/run.txt
  ```
- Additional dependencies using pip will be installed, when running:
  ```shell
  pip install -r requirements/pip_requirements.txt
  ```
Now, all `mti_extract_write_plot` dependencies are now installed for the
`pyechem` conda environment. The user will now be able to run the
`mti_extract_write_plot.py` file:
```shell
python mti_extract_write_plot.py
```
