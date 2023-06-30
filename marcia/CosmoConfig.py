import os
import toml
import numpy as np
import configparser

# 1.) This is anexample file on how to create the config file 
# 2.) We read the config file and create the comsoconfig class
# 3.) We define the functions to read the config file elsewhere in the code 

# Create a configparser object
config = configparser.ConfigParser()

# This is the model name
model = 'LCDM'
sample = False
# To set the name of the file to be written,can be changed to any name however called correctly in the code
filename = 'cosmology.ini'

# If utilising a constant cosmological mean then the following paramters are required
config['GENERAL'] = {'model': model, 'sample': sample}






