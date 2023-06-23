import toml
import os
from numba.experimental import jitclass
from numba import types

current_dir = os.path.dirname(os.path.realpath(__file__))
act_filepath = os.path.join(current_dir,'params.ini')
act_params = toml.load(act_filepath)


specDictToObject = [
    ('dictionary', types.DictType(types.unicode_type, types.Any)),
]
#@jitclass(specDictToObject)
class DictToObject:
    """Converts a dictionary to an object.

    This class uses the __init__ method to convert a dictionary to an object.
    It does so by iterating through the dictionary and setting each key as
    an attribute of the object, with the value of the key being the value of
    the attribute.
    """

    def __init__(self, dictionary: dict):
        """Converts a dictionary to an object.

        Args:
            dictionary (dict): The dictionary to convert to an object.
        """
        for key, value in dictionary.items():
            setattr(self, key, value)

# spec = [
#     ('parameters', types.ListType(types.unicode_type)),
#     ('act_params', types.DictType(types.Any)),
#     ('upd_priors', types.DictType(types.Any)),
# ]

#@jitclass(spec)
class Params:

    def __init__(self,parameters,filepath=None):
        self.act_params = act_params
        if filepath is not None:
            upd_priors = toml.load(filepath)['Priors']
            self.upd_priors = upd_priors
        else:
            self.upd_priors = {}
        
        self.parameters = parameters
        check = self.check_params
    

    def __call__(self,parameters):
        param_dic = {}
        for i,param in enumerate(self.parameters):
            param_dic[param] = parameters[i]
        final_param = dict(param_dic.items() | self.Defaults.items())
        return DictToObject(final_param)
        
    
        
    
    @property
    def check_params(self):
        for param in self.parameters:
            if param not in self.act_params['Label'].keys():
                raise ValueError(f"""
                Parameter '{param}' not defined.
                The available parameters are:
                {list(self.act_params["Label"].keys())}
                """)
        return True



    @property
    def Labels(self):
        label_dict = self.act_params['Label']
        label = []
        for param in self.parameters:
            label.append(label_dict[param])
        return label
    
    @property
    def Priors(self):
        prior_dict = self.act_params['Priors']
        prior = []
        for param in self.parameters:
            if (len(self.upd_priors.keys()) == 0) and (param in self.upd_priors.keys()):
                prior.append(self.upd_priors[param])
            else:
                prior.append(prior_dict[param])
        return prior

    @property
    def Defaults(self):
        default_dict = self.act_params['Defaults']
        default = {}
        for param in default_dict.keys():
            if param in self.parameters:
                continue
            else:
                default[param] = default_dict[param]
        return default
    
