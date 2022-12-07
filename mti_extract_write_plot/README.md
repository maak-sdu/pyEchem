# mti_extract_write_plot

## What does the program do?
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

A number of plots will be created, including:
- time vs. the working electrode potential
- working ion content of the working electrode vs. the working electrode
  potential
- capacity (charge and discharge) vs. working electrode potential
- cycle number vs. capacity (charge and discharge)
- cycle number vs. couloumbic efficiency

All plots will be saved to `png`, `pdf`, and `svg` files.

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
conda environment.

## Running the program
Having all dependencies installed in the `pyechem` conda environment and having
it activated, the user will be able to run the `mti_extract_write_plot.py` file:
```shell
python mti_extract_write_plot.py
```

The user will have to prompt a series of information. Some information is meant
for calculation, while other information is meant for plotting aspects. The user
will have to prompt information for each `.txt` file in the `data` directory.

## Example
Within the `example` folder found in this GitHub repository, a `data` folder
containing an example `.txt` file can be found. In addition to the `data`
folder, a `mti_extract_write_plot.py` file can be found.

Navigate to the `example` directory and run the program:
```shell
python mti_extract_write_plot.py
```

Sample information:
- Mass of activate electrode material: `14.875`
- Empirical formula for working electrode: `LiNi0.8Mn0.1Co0.1O2`
- Initial working ion content of the working electrode: `1`
- Working ion is Li, prompt: `0`
- The counter electrode us LTO, prompt: `4`

The remaining prompts are solely plot-related and the following is just one
possibility:
- Colorbar or legend, for colorbar prompt: `0`
- Colormap type, for user-defined colormap prompt: `0`
- Colormap, for user-defined red colormap prompt: `0`
