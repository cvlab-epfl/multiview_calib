import os
import sys
import json
import re
import os
import ast
import glob
import pickle
import numpy as np

__all__ = ["json_read", "json_write", "pickle_read", "pickle_write", 
           "mkdir", "sort_nicely", "find_files", "invert_Rt",
           "draw_points", "draw_rectangles", "dict_keys_to_string",
           "dict_keys_from_literal_string"]

def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:    
            data = json.load(f)
        return data
    except:
        raise ValueError("Unable to read JSON {}".format(filename))
        
def json_write(filename, data):
    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), 'w') as f:
            json.dump(data, f, indent=2)
    except:
        raise ValueError("Unable to write JSON {}".format(filename))   
        
def pickle_read(filename):
    with open(filename, "rb") as f:    
        data = pickle.load(f)
    return data

def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)        

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def find_files(file_or_folder, hint=None, recursive=False):
    # make sure to use ** in file_or_folder when using recusive
    # ie find_files("folder/**", "*.json", recursive=True)
    import os
    import glob
    if hint is not None:
        file_or_folder = os.path.join(file_or_folder, hint)
    filenames = [f for f in glob.glob(file_or_folder, recursive=recursive)]
    filenames = sort_nicely(filenames)    
    filename_files = []
    for filename in filenames:
        if os.path.isfile(filename):
            filename_files.append(filename)                 
    return filename_files

def dict_keys_to_string(d):
    return {str(key):value for key,value in d.items()}

def dict_keys_from_literal_string(d):
    new_d = {}
    for key,value in d.items():
        if isinstance(key, str):
            try:
                new_key = ast.literal_eval(key)
            except:
                new_key = key
        else:
            new_key = key
        new_d[new_key] = value
    return new_d

def invert_Rt(R, t):
    Ri = R.T
    ti = np.dot(-Ri, t)
    return Ri, ti