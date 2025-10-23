.. highlight:: shell

===========================
Contributing to `OpenTNSim`
===========================

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will
always be given.

This document describes how you can contribute to `OpenTNSim`, the review process, and how your
contribution gets accepted.

Thank you, and happy coding!

Ways to contribute
==================

You can contribute in many ways:

Report bugs
-----------

Report bugs at https://github.com/TUDelft-CITG/OpenTNSim/issues. The issue can be used to track the
progress of the bug fix, and for discussion with maintainers and other developers.

If you are reporting a bug, please use the `Bug` issue template, through which you include:

- A clear and descriptive title.
- Name and version of your operating system, Python version, and `OpenTNSim` version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

Preferably, you also include a minimal code example that reproduces the bug.

Fix bugs
--------

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to
whoever wants to implement it.

_When you fix a bug, please take into account the agreements on developing in `OpenTNSim`, described
below._

Report new ideas
----------------

If you have an idea for a new feature or improvement, cool! Feature requests can be reported at
https://github.com/TUDelft-CITG/OpenTNSim/issues. Please use the `Feature Request` issue template,
through which you include:

- A clear and descriptive title.
- A detailed description of the feature or improvement.
- Any relevant context or examples that might help understand the request.

If you have a code example or prototype, please include that as well.

Implement new features
----------------------

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is
open to whoever wants to implement it.

Please follow the agreements on developing in `OpenTNSim`, described below, and feel free to get
started!

Add examples
------------

Examples are a great way to help users understand how to use `OpenTNSim` and its features. In
`OpenTNSim` examples are provided as jupyter notebooks in the `notebooks` folder. You can contribute
new examples or improve existing ones.

Please follow the agreements on developing in `OpenTNSim`, described below, and feel free to get
started!

Write documentation
-------------------

OpenTNSim could always use more documentation, whether as part of the official OpenTNSim docs, in
docstrings, or even on the web in blog posts, articles, and such.

Did you find something missing in the documentation? For example

- missing docstring in a function, class or module
- docstring that does not corresponds with the actual functionality
- wrongly formatted docstring
- something that could be explained better
- or a missing section in the documentation website

You can contribute by adding or improving documentation.

Please follow the agreements on developing in `OpenTNSim`, described below, and feel free to get started!


Submit feedback
---------------

The best way to send feedback is to file an issue at https://github.com/TUDelft-CITG/OpenTNSim/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)


Agreements on developing in `OpenTNSim`
=======================================

This section outlines the general agreements and guidelines to follow when contributing code to `OpenTNSim`.

In summary:

- any contribution should be made in a feature branch, created from the `develop` branch
- feature branches are short lived
- contributions should be submitted via pull requests to the `develop` branch
- the pull request is used for code review and discussion
- work that is merged into `develop` should meet the Definition of Done (DoD), described below
- commits should be meaningful, small, and have clear messages
- tickets on the issue board should be used to track work and for discussion, their description should be clear and detailed


Definition of Done (DoD)
------------------------

A contribution is considered 'done' when it meets the following criteria:

- The contribution
  - is implemented in a feature branch and submitted via a pull request to the `develop` branch
  - has been reviewed and approved by at least one maintainer or designated reviewer
  - includes tests that cover the new functionality or bug fix, and all tests pass
- The developed code
   - is well-structured, readable, and contains comments where necessary
   - follows the BLACK coding style
   - is tested with `pytest`, and all tests pass
   - did not break existing functionality, where applicable deprecation warnings are added for future breaking changes
- The documentation
  - has been updated to reflect the changes, including docstrings and examples if applicable
  - includes docstrings in `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`__ format
  - includes an update of the release notes in `HISTORY.rst`, if applicable

Only then can a contribution can be merged into the main branch.


Branching
---------

Please follow this branching model when developing in `OpenTNSim`:

main
~~~~

The `main` branch contains the latest released version of `OpenTNSim`. This branch should always be stable and
ready for deployment. No direct commits should be made to this branch. Work in this branch meets the DoD.


develop
~~~~~~~

The `develop` branch contains the latest development code. This is where all completed features and bug fixes
are merged into. No direct commits should be made to this branch. Work in  this branch meets the DoD. 


feature branches
~~~~~~~~~~~~~~~~

Feature branches are created from the `develop` branch for each new feature or bug fix. These branches
are used for active development and testing. Work in these branches does not need to meet the DoD until it is
ready to be merged into the `develop` branch via a pull request.

Feature branches are short lived and should be deleted after the work is merged into `develop`.


Pull requests
-------------

Pull requests (PRs) are the primary way to submit changes to the `OpenTNSim` project. When you are ready 
to merge your feature branch into the `develop` branch, create a pull request on GitHub.

Make sure to include a clear description of your changes and any relevant context. This will help
reviewers understand your work and provide feedback.


Commit history
--------------

When coding in `OpenTNSim`, please make sure that your commit history is clean and
meaningful. This helps maintainers understand the changes you made and their purpose.

Take into account the following guidelines when making commits:

* Use clear and descriptive commit messages.
* Break your changes into small, manageable commits.
* Include tests for your changes.
* Follow the coding style and conventions used in the project.



Review process
--------------

The review process for contributions to `OpenTNSim` is as follows:

- the contributor creates a pull request (PR) from their feature branch to the `develop` branch
- a reviewer is assigned to the PR, either by the contributor or a maintainer
- the reviewer reviews the code by testing it, and checking that the DoD is met
- the reviewer provides feedback, which may include requests for changes or improvements
- the contributor addresses the feedback and makes any necessary changes, repeat until the reviewer is satisfied
- once the reviewer approves the PR, a maintainer merges the changes into the `develop` branch


Release policy
--------------

`OpenTNSim` follows semantic versioning for its releases, i.e. versions are numbered as <major>.<minor>.<patch>. The release 
policy is as follows

- Small additions, bug fixes, and documentation updates are released as patch releases. Patch releases should be lightweight.
- Larger new features, enhancements, or refactoring that do not break backward compatibility are released as minor releases.
- Major upgrades, significant new features, or refactoring that may break backward compatibility are released as major releases.


Get Started!
============

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



