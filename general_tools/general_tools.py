# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python [conda env:oggm_env]
#     language: python
#     name: conda-env-oggm_env-py
# ---

# # Imports

import os
import shutil


# # Notebook or script?
#
# If we developing or debugging the code in the notebook we use the is_notebook, if executed as a script on the cluster we ignore everything related to is_notebook.
#
# - You can connect the script and the notebook automatically using jupytext https://jupytext.readthedocs.io/en/latest/install.html.
# - Or, whenever something is changed, you must 'Save and Export Notebook As...' 'Executeable Script'.

# Function to detect if we're running in a Jupyter notebook
def check_if_notebook():
    try:
        shell_name = get_ipython().__class__.__name__
        if shell_name == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or JupyterLab
        elif shell_name in ['TerminalInteractiveShell', 'InteractiveShell']:
            return False  # IPython terminal or other interactive shells
        else:
            # Fallback or default behavior for unidentified environments
            return False
    except NameError:
        return False      # Not in IPython, likely standard Python interpreter


# # # mkdir adapted from OGGM utils

def mkdir(path, reset=False):
    """Checks if directory exists and if not, create one.

    Parameters
    ----------
    reset: erase the content of the directory if exists

    Returns
    -------
    the path
    """

    if reset and os.path.exists(path):
        shutil.rmtree(path)
        # deleting stuff takes time
        while os.path.exists(path):  # check if it still exists
            pass
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return path
