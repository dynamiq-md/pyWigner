import numpy as np

def clean_ravel(arr, n_dofs):
    try:
        n_dim = len(arr)
        retval = arr.ravel()
    except TypeError: # if len(arr) throws error
        retval = [arr] * n_dofs
    except AttributeError: # arr.ravel() throws error but len(arr) doesn't
        retval = arr
    return retval


