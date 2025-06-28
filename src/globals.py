'''
globals.py
5/9/2025
Harry Lynch
Holds definitions for variables used across modules in the program- namely the
1-hot associations of each label string as they appear in the dataset and the 
label ct.
'''

# Probably a better way to do this but this was fast- each dict entry maps to a 1-hot
# encoding, so Iris-setosa=[1,0,0], etc.
LABELS_TO_IDX={"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
IDX_TO_LABELS={0:"Iris-setosa", 1:"Iris-versicolor", 2:"Iris-virginica"}
NUM_CLASSES=3