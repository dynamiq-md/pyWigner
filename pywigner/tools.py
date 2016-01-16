def clean_ravel(arr, n_dofs):
    try:
        retval = arr.ravel()
    except AttributeError:
        retval = [arr] * n_dofs
    return retval

