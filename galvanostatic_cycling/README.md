# galvanostatic_cycling

## What does the program do?
The code found within the `galvanostatic_cycling.py` file will extract data
exported to a `.txt` file from the Biologic or MTI potentiostats used at the
Department of Chemistry, Aarhus University.

The program will process `.txt` files found in a `data` folder in the
`galvanostatic_cycling` directory, i.e. `galvanostatic_cycling/data`.

When running the program, `txt`, `png`, `pdf`, and `svg` folders will be created
for output files. For each of the `.txt` files in the `data` directory, a
subfolder with the name of the `.txt` file will be created for each of the
output directories.

For the case of `.txt` files containing Biologic data, `data_comma-to-dot` and
`data_no-header` folders, and `.txt` files will be written to these. However,
these should only be considered as modified inputs files and not output files.

The time (t), working electrode potential (Ewe), current (I), capacity (Q), and
working ion content of the working electrode (x) will be exported to more
user-friendly `.txt` files. A file containing all the data will be written as
well as a file for each cycle.

A number of plots will be created, including:
- time vs. the working electrode potential (t vs. Ewe)
- working ion content of the working electrode vs. the working electrode
  potential (x vs. Ewe)
- capacity (charge and discharge) vs. working electrode potential (Q vs. Ewe)
- cycle number vs. capacity, both charge and discharge (cycles number vs. Q)
- cycle number vs. couloumbic efficiency (cycle number vs. CE)

All plots will be saved to `png`, `pdf`, and `svg` files.

## Running the program
Having all dependencies installed in the `pyechem` conda environment and having
it activated, the user will be able to run the `galvanostatic_cycling.py` file:
```shell
python galvanostatic_cycling.py
```

Alternatively, the program can be run as an iPython notebook, `.ipynb` file.
To run the `.ipynb` file, initiate a Jupyter Notebook or Jupyter Lab session:
```shell
jupyter notebook
```

or

```shell
jupyter-lab
```

The user will have to prompt a series of information. Some information is meant
for calculation, while other information is meant for plotting aspects. The user
will have to prompt information for each `.txt` file in the `data` directory.
Please see the examples below.

## Examples
Within the `example` folder found in this GitHub repository, a `data` folder
containing an example `.txt` file can be found. In addition to the `data`
folder, a `mti_extract_write_plot.py` file can be found.

Navigate to the `example` directory and run the program:
```shell
python mti_extract_write_plot.py
```
### Example 1: `00_BIOLOGIC_NO_CURRENT.txt`
This file contains Biologic data. However, as the current (I) is missing from
the file, this file will be skipped.

### Example 2: `01_BIOLOGIC_NO_X.txt`
This file contains Biologic data. However, as the working ion content (x) is
missing from the file, this file will be skipped.

### Example 3: `02_BIOLOGIC_Na07Fe033Mn067O2-Na_6dot29.txt`
This file contains Biologic data. As the data were exported from EC-Lab without
a header containing metadata, the user will have to provide the mass of the
active electrode material.

Sample information:
- Type of potentiostat, for Biologic prompt: `0`
- Mass of activate electrode material (mg): `6.29`
- Empirical formula for working electrode: `Na0.7Fe0.33Mn0.67O2`
- Initial working ion content of the working electrode: `0.7`
- Working ion is Na, prompt: `1`
- The counter electrode is Na, prompt: `1`

The remaining prompts are solely plot-related and the following is just one
possibility:
- Colorbar or legend, for colorbar prompt: `0`
- Colormap type, for user-defined colormap prompt: `0`
- Colormap, for user-defined red colormap prompt: `0`

### Example 4: `02_BIOLOGIC_Na07Fe033Mn067O2-Na_6dot29.txt`
This file contains Biologic data identical to Example 3. However, here, the data
were exported from EC-Lab with a header containing metadata, such that the user
does not have to provide the mass of the active electrode material.

Sample information:
- Type of potentiostat, for Biologic prompt: `0`
- Empirical formula for working electrode: `Na0.7Fe0.33Mn0.67O2`
- Initial working ion content of the working electrode: `0.7`
- Working ion is Na, prompt: `1`
- The counter electrode is Na, prompt: `1`

The remaining prompts are solely plot-related and the following is just one
possibility:
- Colorbar or legend, for colorbar prompt: `0`
- Colormap type, for user-defined colormap prompt: `0`
- Colormap, for user-defined red colormap prompt: `0`

### Example 5: `10_MTI_LiNMC811-LTO_14dot875.txt`
This file contains MTI data.

Sample information:
- Type of potentiostat, for MTI prompt: `1`
- Mass of activate electrode material (mg): `14.875`
- Empirical formula for working electrode: `LiNi0.8Mn0.1Co0.1O2`
- Initial working ion content of the working electrode: `1`
- Working ion is Li, prompt: `0`
- The counter electrode is LTO, prompt: `4`

The remaining prompts are solely plot-related and the following is just one
possibility:
- Colorbar or legend, for colorbar prompt: `0`
- Colormap type, for user-defined colormap prompt: `0`
- Colormap, for user-defined red colormap prompt: `0`
