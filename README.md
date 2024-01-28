# MRI_Py
Python program for continuous segmentation of MRI data

This program can be used for the continuous segmentation of subjects' MRI data, it has been used in "Continuous Cortical Gray-/White-Matter Boundary Segmentation and Analysis" written by me (Björn van Amstel).
Dependencies are Nighres, Scikit-Learn, NumPy, NiBabel, SciPy, and MatplotLib.

The workflow is as follows: Place all subject MRI's in a folder, change the root variable into the path for this folder, specify the desired ROI, specify the desired number of clusters to be generated and other K-Means parameters, and run the program.
