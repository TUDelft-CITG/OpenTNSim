.. highlight:: shell

============
Installation
============


Stable release
--------------

To install OpenTNSim, run this command in your terminal:

.. code-block:: bash

    # Use pip to install OpenTNSim
    pip install opentnsim

This is the preferred method to install OpenTNSim, as it will always install the most recent stable release.

If you do not `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for OpenTNSim can be downloaded from the `Github repo`_.

You can either clone the public repository via git:

.. code-block:: bash

    # Use git to clone OpenTNSim
    git clone https://github.com/TUDelft-CITG/OpenTNSim.git


If you don't have git installed, you can download the `tarball`_ via curl and unzip. Make sure curl is installed on your system:

.. code-block:: bash

    # Use curl to obtain the tarball and unzip it
    curl  -L https://github.com/TUDelft-CITG/OpenTNSim/tarball/master | tar -xz


Once you have a copy of the source, you need to create a virtual environment to install the packages in. Run the following code in the base directory of the OpenTNSim folder:

.. code-block:: bash

    # Create a virtual environment
    python3 -m venv .venv 


Now you can install the package using pip:

.. code-block:: bash

    # Create a virtual environment (the dot is important!)
    pip install -e .
    pip install -e .[testing]

.. _Github repo: https://github.com/TUDelft-CITG/OpenTNSim
.. _tarball: https://github.com/TUDelft-CITG/OpenTNSim/tarball/master
