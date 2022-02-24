# EC-Lab extractor
This code can extract and plot desired columns exported from the EC-lab software
to .txt files. EC-lab is the software that comes with Biologic potentiostats
used in the Ravnsbæk Group.

As EC-lab might export .txt files using comma as decimal separator, the code
also converts commas to dots. From these files, columns desired by the user can
be extracted to two-column .txt files and plots will be saved as .png and .pdf
files.

If initial resting is part of the electrochemical experiment, the initial
resting can be excluded, if wished so.
