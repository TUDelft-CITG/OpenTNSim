import opentnsim

def find_notebook_path():
    """Lookup the path where the notebooks are located. Returns a pathlib.Path object."""
    opentnsim_path = pathlib.Path(opentnsim.__file__)
    # check if the path looks normal
    assert 'opentnsim' in str(opentnsim_path), "we can't find the opentnsim path: {opentnsim_path} (opentnsim not in path name)"
    # src_dir/opentnsim/__init__.py -> ../.. -> src_dir
    src_path = opentnsim_path.parent.parent
    notebook_path = opentnsim_path.parent.parent / "notebooks"
    return notebook_path
