# mti_extract_write_plot
The code found within the `mti_extract_write_plot.py` file will extract data
exported to a `.txt` file from the MTI potentiostat used at the Department of
Chemistry, Aarhus University.

The program will process `.txt` files found in a `data` folder in the
`mti_extract_write_plot` directory, i.e. `mti_extract_write_plot/data`.

When running the program, `txt`, `png`, `pdf`, and `svg` folders will be created
for output files. For each of the `.txt` files in the `data` directory, a
subfolder with the name of the `.txt` file will be created.

The time, working electrode potential, current, capacity, and working ion
content of the working electrode will be exported to more user-friendly `.txt`
files. This goes for all of the data but also for each individual cycle.

A number of plots will be created, including time vs. the working ion potential,
working ion content of the working electrode vs. the working electrode
potential, capacity (charge and discharge) vs. working electrode potential,
cycle number vs. capacity (charge and discharge), and cycle number vs.
couloumbic efficiency. All plots will be saved to `png`, `pdf`, and `svg`
files.

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
- Using pip, additional dependencies will be installed, when running:
  ```shell
  pip install -r requirements/pip_requirements.txt
  ```
Now, all `mti_extract_write_plot` dependencies are installed for the `pyechem`
conda environment. The user will now be able to run the
`mti_extract_write_plot.py` file:
```shell
python mti_extract_write_plot.py
```
