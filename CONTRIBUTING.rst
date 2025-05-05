.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/TUDelft-CITG/OpenTNSim/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

OpenTNSim could always use more documentation, whether as part of the
official OpenTNSim docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/TUDelft-CITG/OpenTNSim/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `OpenTNSim` for local development.

1. Fork the `OpenTNSim` repository on GitHub.


2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/OpenTNSim.git


3. Install your local copy into a virtualenv. Assuming you have pip installed, this is how you set up your fork for local development::

    $ cd opentnsim/
    $ pip install -e
    $ pip install -e[testing]


4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature


   Now you can make your changes locally.

5. Make sure your changes are tested. Make a test for your changes in the 'tests'-folder. 
Make sure the name of the file starts with 'test_' and the name of the test function starts with 'test_' as well. 
This is important for pytest to find your tests.

 OpenTNSim uses pytest for testing. You can run all tests using::

    $ pytest


 or run a specific file with tests using::
   
    $ pytest tests/<python_file>.py

6. If you add new functionality, add a jupyter notebook on how to us this feature. save the notebook in the 'notebooks' folder. 
   Use example 00 - Basic simulation as an example for the layout of your notebook.


7. The style of OpenTNSim is according to Black. When you're done making changes, format your code using 
   Black with the following lines of code::

    $ black opentnsim tests


   You can install black using pip.

8. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 opentnsim tests
    $ pytest
    $ tox


   To get flake8 and tox, just pip install them into your virtualenv.

9. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature


10. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add a notebook with an example on how to use the feature.
3. The pull request should work for all python versions in the bugfix-phase, see https://devguide.python.org/versions/. Check
   CircleCI and make sure that the tests pass.

Tips
----

To run a subset of tests::

$ pytest tests.test_opentnsim

To make the documentation pages
$ make docs # for linux/osx. Not supported for windows

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags

Travis will then deploy to PyPI if tests pass.
