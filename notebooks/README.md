# Overview of notebooks
These are the notebooks for the OpenTNSim package.

# Cleanup
archive
*Freder* -> work Frederik Vinke
*Tijmen* -> student of Frederik
*Difference maps* -> Work of Loes
Energy Consumption -> Mark
OD*.xlsx -> (Loes, check input/output -> move to students)
*Locking* -> Floor
*Graph.pickle* -> Loes
Iterator -> fedor Remove
robert -> Man Jiang
Examples -> Mark, Combine with other notebook

# Intended structure

# static notebooks (with old versions opentnsim)
notebooks/papers (test branch/unsbumitted papers)
notebooks/papers/jiang2022 # version OpenTNSim 1.1
notebooks/papers/jiang2022/figure-3-5.ipynb
notebooks/papers/jiang2022/requirements.txt
notebooks/data/shape-files

notebooks # Main notebooks, describe and explain feature/functionality (test + book)
notebooks/basic-move-energy.ipynb
notebooks/basic-move-with-energy.ipynb

archive -> things that will be removed

book?/toc.yml (chapter 2: basic-move-with-energy.ipynb)
doc/index.toc


# test single features
tests/test-feature.py
opentnsim/core.py
opentnsim/energy.py


# Files
All result files should be removed. These can be created as artifacts in the circleci environmet. Larger input files should be moved to zenodo.
