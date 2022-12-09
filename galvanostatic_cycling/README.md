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
