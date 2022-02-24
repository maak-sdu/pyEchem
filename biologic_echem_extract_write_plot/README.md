# EC-Lab extractor
This code will extract desired columns from exported .txt files from the EC-lab
software that comes with Biologic potentiostats used in the Ravnsbæk Group. The
extracted columns will be written to two-column .txt files and plotted as .png
and .pdf files.

As EC-lab might export .txt files using comma as decimal separator, the code
also converts commas to dots. From these files, columns desired by the user can
be extracted to two-column .txt files and plots will be saved as .png and .pdf
files.

If initial resting is part of the electrochemical experiment, the initial
resting can be excluded, if wished so.
