# pyEchem
Processing, analyzing, and plotting electrochemical data.

The code in this repository is directed towards the electrochemical data
encountered in the Ravnsbæk Group at Aarhus University.

The current types of electrochemical experiments include galvanostatic cycling
(GC), cyclic voltammetry (CV), and galvanostatic intermittent titration
technique (GITT).

Current potentiostats (battery cyclers) include those from Biologic, MTI, and
Maccor.

## Setup and installation
The following guidelines assume that the user runs a conda distribution, i.e.,
Anaconda or Miniconda. If these guidelines are followed, all dependencies for
the `pyEchem` code will be installed.

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
- Navigate to the main `pyEchem` directory. Using conda, dependencies present in
  the `run.txt` file in the `requirements` directory will be installed from the
  `conda-forge` channel, when running:
  ```shell
  conda install -c conda-forge --file requirements/run.txt
  ```
- Using pip, additional dependencies present in the `pip_requirements.txt` file
  in the `requirements` directory will be installed, when running:
  ```shell
  pip install -r requirements/pip_requirements.txt
  ```
- If you are running on a Windows OS, run:
  ```shell
  pip install -r requirements/pip_windows.txt
  ```
- If you are running on a Unix OS (Mac OS or Linux), run:
  ```shell
  conda install -c conda-forge --file requirements/conda_unix.txt
  ```
Now, all `pyEchem` dependencies are installed for the `pyechem` conda
environment. You are now ready to run the code present in the `pyEchem`
repository.
