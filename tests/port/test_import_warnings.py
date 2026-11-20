"""Importing the port module must not warn about pandas options (pandas 3 deprecated mode.chained_assignment)."""

import subprocess
import sys


def test_importing_the_port_utils_gives_no_pandas_option_warning():
    # in a new process: the module is imported once per process, so the warning is only given at the first import
    command = [
        sys.executable,
        "-W",
        "error:The 'mode.chained_assignment' option is deprecated",
        "-c",
        "import opentnsim.port.utils",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
