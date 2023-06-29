import os
import toml
import numpy as np
import configparser

# 1.) This is anexample file on how to create the config file 
# 2.) We read the config file and create the GPconfig class
# 3.) We define the functions to read teh config file elsewhere in the code 

# Create a configparser object
config = configparser.ConfigParser()


# Set the number of tasks
n_Tasks = 2

# To set the name of the file to be written,can be changed to any name however called correctly in the code
filename = 'GPconfig.ini'

# Set the number of tasks in the GENERAL section
# self scale implies all the tasks have the same length scale paramter. 
config['GENERAL'] = {'n_Tasks': n_Tasks, 'self_scale': True} 

config['KERNEL'] = {'Interpolate': True, 'Method': 'cubic', 'n_points': 100 }


# Set the configuration for each task: Need not be a loop, can be done manually as well 
for i in range(n_Tasks):
    Task = 'Task_' + str(i+1)
    config.add_section(Task)
    config.set(Task, 'model', 'SE')
    # Set the model nu value, relevant only for the Matern kernel 
    config.set(Task, 'nu', '0.0')
    # Set the hyperparameter ranges
    config.set(Task, 'l_min', '0.001')
    config.set(Task, 'l_max', '10.0')
    config.set(Task, 'sigma_f_min', '0.001')
    config.set(Task, 'sigma_f_max', '10.0')

# if necessary to include a intrinsic scatter or an offset to the covariance matrix 
config['INTRINSIC_SCATTER'] = {'sigma_int': True, 'offset': True}
if config['INTRINSIC_SCATTER']['sigma_int']:
    config['INTRINSIC_SCATTER']['sigma_int'] = '0.1'
if config['INTRINSIC_SCATTER']['offset']:
    config['INTRINSIC_SCATTER']['offset'] = '0.0001'

# Write the configparser object to a file
with open(filename, 'w') as configfile:
    config.write(configfile)
           
# Read the config file and create the GPparams class
class GPConfig:
    """
    Class to read the config file and create the GPparams class
    """
    def __init__(self, filename):
        """
        Constructor to read the config file and create the GPparams class
        """
        # Read the config file
        config = configparser.ConfigParser()
        config.read(filename)
        self.models = []
        self.nus = []

        # Set the number of tasks
        self.n_Tasks = config.getint('GENERAL', 'n_Tasks')
        
        # Set the self_scale parameter
        self.self_scale = config.getboolean('GENERAL', 'self_scale')

        # Set the interpolation parameters
        self.Interpolate = config.getboolean('KERNEL', 'Interpolate')
        self.Method = config.get('KERNEL', 'Method')
        self.n_points = config.getint('KERNEL', 'n_points')

        # Set the configuration for each task
        for i in range(self.n_Tasks):
            Task = 'Task_' + str(i+1)
            # Set the model
            model = config.get(Task, 'model')
            self.models.append(model)
            # Set the model nu value, relevant only for the Matern kernel
            nu = config.getfloat(Task, 'nu')
            self.nus.append(nu) 
            # Set the hyperparameter ranges
            l_min = config.getfloat(Task, 'l_min')
            l_max = config.getfloat(Task, 'l_max')
            sigma_f_min = config.getfloat(Task, 'sigma_f_min')
            sigma_f_max = config.getfloat(Task, 'sigma_f_max')
            # Create the GPparams class
            self.__dict__[Task] = {'model': model, 'nu': nu, 'l_min': l_min, 'l_max': l_max, 'sigma_f_min': sigma_f_min, 'sigma_f_max': sigma_f_max}
        
        
