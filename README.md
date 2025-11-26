[![Documentation](https://img.shields.io/badge/sphinx-documentation-informational.svg)](https://opentnsim.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](https://github.com/TUDelft-CITG/Transport-Network-Analysis/blob/master/LICENSE.txt)
[![DOI](https://zenodo.org/badge/145843547.svg)](https://zenodo.org/badge/latestdoi/145843547)

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/TUDelft-CITG/OpenTNSim/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/TUDelft-CITG/OpenTNSim/tree/master)

# OpenTNSim

**Open** source **T**ransport **N**etwork **Sim**ulation -  Analysis of traffic behaviour on networks for different traffic scenarios and network configurations.

Documentation can be found: [here](https://opentnsim.readthedocs.io/)

## Book

<a href="https://happy-bush-0c5d10603.1.azurestaticapps.net"><img src="docs/_static/book.png" style="max-width: 50vw;"></a>

You can find the opentnsim book, based on the examples in the `notebooks` folder on the [opentnsim-book](https://happy-bush-0c5d10603.1.azurestaticapps.net/) website.

## Installation

### Quick Start

**Prerequisites:**
- Python 3.12 or higher
- pip
- Dependency manager: Poetry or uv (Only need one of them)

If you do not have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process. You can read the [documentation](https://opentnsim.readthedocs.io/en/latest/installation.html) for other installation methods and a more detailed description. 

It is recommended to use either Poetry or uv for as dependency manager. For help on installing, using and trouble shooting these dependencymanagers see `Virtual Environments basics` folder on  [opentnsim-book](https://happy-bush-0c5d10603.1.azurestaticapps.net/).

#### User 
To install OpenTNSim, run this command in your terminal:

``` bash
pip install opentnsim
```

Then also install the extra dependencies used for testing and notebook support:
``` bash
pip install opentnsim[testing]
```

#### Local Development: Contributing & Notebooks
If you want to contribute or run notebooks follow the steps below:

**Using uv**
install packages.This will also automatically create your virtual environment under .venv
``` bash
uv sync --extra testing
```
**Using Poetry**
``` bash
poetry install --extras testing
```

Activate your virtual environment:
``` bash
# Linux
source .venv/bin/activate

# Windows
source .venv\Scripts\activate
```



You can now run scripts or tests:
```bash
## run specific python script
python script.py

# run all tests
pytest

# run single tests
pytest -k test_graph

# run notebooks tests
pytest --nbmake ./notebooks --nbmake-kernel=python3 --ignore ./notebooks/cleanup --ignore ./notebooks/student_notebooks --ignore ./notebooks/broken

# run with coverage
pytest --cov=opentnsim
```
### Working with Jupyter Notebooks

1. Check if Jupyter extension is installed with  `poetry show ipykernel` or `uv pip show ipykernel`.
2. Open any `.ipynb` file
3. Select "opentnsim" kernel. Should look something like: `opentnsim-py3.13`.

## Examples
The benefit of OpenTNSim is the generic set-up. A number of examples are presented in in the `notebooks` folder on the [opentnsim-book](https://happy-bush-0c5d10603.1.azurestaticapps.net/) website. Additional examples can be found in the notebooks-folder in this repository. 

## Book

Based on the examples and docs a book can be generated using the commands `make book` and cleaned up using `make clean-book`. These commands are unix only.

## Code quality
Code quality is checked using sonarcloud. You can see results on the [sonarcloud](https://sonarcloud.io/project/overview?id=TUDelft-CITG_OpenTNSim) website. For now we have disabled coverage and duplication checks. These can be enabled when we include coverage measurements and reduce duplication by optimizing the tests.





