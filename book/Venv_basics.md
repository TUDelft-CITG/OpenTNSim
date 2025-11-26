# Tutorial - Set up Environments

The following section explains how you get from nothing to launching Jupyter Notebook through Anaconda. The explaination below is supported by two videos showing how to do certain steps in Windows. The following Youtube playlists visualize the key steps of this tutorial:

**Windows**

- https://www.youtube.com/watch?v=QLDm2aQcyG8&list=PLQ6E4F0YWxJn3-lGxFg6j7IkShzy69WDc

**OSX**

- https://www.youtube.com/watch?v=_0gQh7JDHIE&list=PLQ6E4F0YWxJkOzPUq0XavFBZfFksrtXDX

### Virtual environments
An environment is a directory that contains a specific collection of packages that you have installed. 

The advantages of using virtual environments are: 
- Prevent dependency issues by allowing you to use different versions of a package for different projects. For example, you could use Package A v2.7 for Project X and Package A v1.3 for Project Y.
- Make your project self-contained and reproducible by capturing all package dependencies in a requirements file.
- Install packages on a host on which you do not have admin privileges.
- Keep your global directory tidy because you don't have to install packages system-wide that you only need for a single project.

### Poetry
Poetry is a dependency manager that automatically handles virtual environments and packages. The steps to install poetry are outlined in the guide  below. For additional information on how to install see the [official Poetry installation guide](https://python-poetry.org/docs/#installing-with-the-official-installer). 

#### Poetry installation guide

**Step 1: Install Poetry**
Installing Poetry via command line:

**Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```
**Linux/Mac:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Verify (Linux/Mac/Windows):** `poetry --version` should show version 2.0.0 or higher.
```bash
Poetry (version 2.2.1)
```
**Upgrade poetry itself**
If it does not show poetry version 2 or higher then update poetry with:
```bash
poetry self update
```
**Step 2: Poetry for OpenTNSim**

```bash
# Clone the repository
git clone https://github.com/TUDelft-CITG/OpenTNSim.git
cd OpenTNSim
```
Install all dependencies (may take some time)
```bash
poetry install 
```

If you want to use notebooks and run tests, then also install the extras
```bash
poetry install --extras testing
```
**Step 3: Verify Installation**
```bash
# Check packages are installed
poetry show

# Test OpenTNSim works
poetry run python -c "import opentnsim; print('opentnsim.__version__')"
```
### uv
Poetry is a dependency manager that automatically handles virtual environments and packages. The steps to install uv are outlined in the guide  below. For additional information on how to install see the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/). 

#### uv installation guide

**Step 1: Install uv**
Installing uv via command line:

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
**Linux/Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Verify (Linux/Mac/Windows):** `uv --version` should show version 0.9.12 or higher.
```bash
uv 0.9.12
```
**Upgrade uv itself**
```bash
uv self update
```
**Step 2: uv for OpenTNSim**

```bash
# Clone the repository
git clone https://github.com/TUDelft-CITG/OpenTNSim.git
cd OpenTNSim
```
Install all dependencies (may take some time)
```bash
uv sync 
```

If you want to use notebooks and run tests, then also install the extras
```bash
uv sync --extra testing
```
**Step 3: Verify Installation**
```bash
# Check packages are installed
uv pip list

# Test OpenTNSim works
uv run python -c "import opentnsim; print('opentnsim.__version__')"

```


### Using Poetry or uv Commands
| Task | Poetry Command | UV Command |
|------|----------------|------------|
| Install dependencies | `poetry install` | `uv sync` |
| Add a package | `poetry add package-name` | `uv add package-name` |
| Remove a package | `poetry remove package-name` | `uv remove package-name` |
| Update packages | `poetry update` | `uv lock --upgrade` |
| List packages | `poetry show` | `uv pip list` |
| Run Python script | `poetry run script.py` | `uv run script.py` |
| Run tests | `poetry run pytest` | `uv run pytest` |
| Activate environment | `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows) | `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows) |

*NOTE*: When you have activated your virtual environment there is no need to include `poetry` or `uv` before running your command. You can run your command as is. e.g: `pyest` or
`pytest -k test_graph`

### Conda 
Conda is a package manager and environment manager that you use with command line commands at the Anaconda Prompt for Windows, or in a terminal window for macOS or Linux (which is used in this tutorial). Further, Anaconda Navigator is a graphical interface to use conda (not considered in this tutorial). 

You have to install Anaconda via https://docs.anaconda.com/anaconda/install/
Once installed, in Windows, go to the Start menu, search for and open "Anaconda Prompt." In MacOS or Linux, open a terminal window. 

A few basic commands are: 
- `conda info`: verify that conda is installed and check the installed version
- `conda update conda`: update to the current version of conda
- `conda --version`: check installed version of conda

When you begin using conda, you already have a default environment named `base`. You don't want to put programs into your base environment, though. Create separate environments to keep your programs isolated from each other.


<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/QLDm2aQcyG8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Creating and checking environments

- To create a new environment named `opentnsimenv`, that uses python version 3.5 specifically (This is just an example, if you leave out `python=3.5` conda installs the same Python version you used when you downloaded and installed Anaconda. This is recommended when you've just downloaded Anaconda):

`conda create --name opentnsimenv python=3.5`

- To activate the environment named `opentnsimenv`:

`conda activate opentnsimenv`

- Verify which version of Python is in your current environment:

`python --version`

- To deactivate the environment and return to the `base` environment:

`conda activate`

- To display a list of all environments (the artrisk* indicates the active environment):

`conda info --envs`


<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/hSh256fmIHY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


### Installing packages in environments

- To check the installed packages within the activated environment:

`conda list`

- To install a particular package in the activated environment, in this case the package `numpy`:

`conda install numpy`

- To install a list of packages in a requirements file, navigate to the file directory and install them: 

`cd [local directory]\notebooks\`

`pip install -r requirements.txt`

### Creating an environment including specific packages

- To create an environment using an `environment.yml` file: 

`conda env create -f environment.yml`

- To create a new environment `copied_env` by copying an existing environment called `original_env`:  

`conda create --name copied_env --clone original_env`

- To create a new environment by copying an existing environment using a text file `spec-file.txt` (this can be used to create the environment on another computer): 

`conda list --explicit > spec-file.txt`

`conda create --name copied_env --file spec-file.txt`

### Activating an environment and starting Jupyter Notebook

- To work on notebooks in an environment name `opentnsimenv`, after just opening the Anaconda Prompt, first activate the `opentnsimenv` environment:

`conda activate opentnsimenv`

- Navigate to the correct directory in which you want to save your work. Save yourself some work by copying the correct directory in the Windows file explorer (Google this if you don't know how) and pasting it after `cd`:

`cd C:/Users/correct/directory/...`

- When in the correct directory, `Jupyter Notebook` can be booted up by:

`jupyter notebook`

### Mamba (instead of Conda)

In case you notice that your computer is slow when using Anaconda, it could be beneficial to install Mamba. Mamba is package manager which can be used with Conda simultaneously. The link listed below takes you to a page where everything from installation to advanced usage of Mamba is explained.<br>


- https://mamba.readthedocs.io/en/latest/index.html




