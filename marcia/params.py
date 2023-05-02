import toml
import os


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

class Params:

    def __init__(self,parameters,filepath=None):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        act_filepath = os.path.join(current_dir,'params.ini')
        self.act_params = toml.load(act_filepath)
        if filepath is not None:
            self.upd_priors = toml.load(filepath)['Priors']
        else:
            self.upd_priors = None
        
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
            if (self.upd_priors is not None) and (param in self.upd_priors.keys()):
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
    
