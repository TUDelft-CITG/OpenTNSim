[![Documentation](https://img.shields.io/badge/sphinx-documentation-informational.svg)](https://opentnsim.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](https://github.com/TUDelft-CITG/Transport-Network-Analysis/blob/master/LICENSE.txt)
[![DOI](https://zenodo.org/badge/145843547.svg)](https://zenodo.org/badge/latestdoi/145843547)

[![TUDelft-CITG](https://circleci.com/gh/TUDelft-CITG/OpenTNSim.svg?style=shield&circle-token=59b1f167ed771129459d86e822fd2faaae8f4a34)](https://circleci.com/gh/TUDelft-CITG/OpenTNSim)

# OpenTNSim

**Open** source **T**ransport **N**etwork **Sim**ulation -  Analysis of traffic behaviour on networks for different traffic scenarios and network configurations.

Documentation can be found: [here](https://opentnsim.readthedocs.io/)

## Book

<a href="https://happy-bush-0c5d10603.1.azurestaticapps.net"><img src="docs/_static/book.png" style="max-width: 50vw;"></a>

You can find the opentnsim book, based on the examples in the `notebooks` folder on the [opentnsim-book](https://happy-bush-0c5d10603.1.azurestaticapps.net/) website.


## Installation
### using the package
To install OpenTNSim, run this command in your terminal:

``` bash
pip install opentnsim
```

To also install the extra dependencies used for testing you can use:
``` bash
pip install opentnsim[testing]
```

This is the preferred method to install OpenTNSim, as it will always install the most recent stable release.

If you don not have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process. You can read the [documentation](https://opentnsim.readthedocs.io/en/latest/installation.html) for other installation methods and a more detailed description.

### local development

The sources for OpenTNSim can be downloaded from the Github repo. You can clone the public repository:

``` bash
# Use git to clone OpenTNSim
git clone git://github.com/TUDelft-CITG/OpenTNSim
```

Once you have a copy of the source, you need to create a virtual environment to install the packages in. Run the following code in the base directory of the OpenTNSim folder:

``` bash
# create virtual environment
python3 -m venv .venv

# install packages (the dot is important!)
pip install -e .
pip install -e .[testing]
```



## Testing
When you have installed the package loacally, you can run the unit tests

```bash
pytest
```

Or you can run the notebook tests:
```bash
pytest --nbmake ./notebooks --nbmake-kernel=python3 --ignore ./notebooks/cleanup --ignore ./notebooks/student_notebooks --ignore ./notebooks/broken
```

Or you can run a specific test like this:

``` bash
pytest -k test_single_engine
```


## Examples

The benefit of OpenTNSim is the generic set-up. A number of examples are presented in in the `notebooks` folder on the [opentnsim-book](https://happy-bush-0c5d10603.1.azurestaticapps.net/) website. Additional examples can be found in the notebooks-folder in this repository. 

## Book

Based on the examples and docs a book can be generated using the commands `make book` and cleaned up using `make clean-book`. These commands are unix only.

## Code quality
Code quality is checked using sonarcloud. You can see results on the [sonarcloud](https://sonarcloud.io/project/overview?id=TUDelft-CITG_OpenTNSim) website. For now we have disabled coverage and duplication checks. These can be enabled when we include coverage measurements and reduce duplication by optimizing the tests.


## OpenCLSim 
OpenTNSim makes use of the [OpenCLSim](https://github.com/TUDelft-CITG/OpenCLSim) code. Both packages are maintained by the same team of developers. You can use these packages together, and combine mixins from both packages. When you experience a problem with integrating the two packages, please let us know. We are working towards further integrating these two software packages.
