# ec-lab_txt_merge

## Merging .txt files containing e-chem data from EC-lab.
The code in this iPython notebook will merge data in .txt files exported from 
the EC-Lab software.

The EC-Lab software will generate one .mpr file for each `technique` used.
If the same technique is used multiple times, e.g., the same experiment but with
different current rates, one may end up one experiment in multiple .mpr files.

When exporting the data, the data will end up in multiple .txt files, which can
be merge using this iPython notebook.
