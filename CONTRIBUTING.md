# Contributing to BeyondML
Thank you very much for your interest in contributing to the BeyondML project! We want to make contributing to this project as easy and transparent as possible, whether you wish to:

- Report a bug
- Discuss the current state of the code
- Submit a fix
- Propose new features
- Become a maintainer

## We Develop with GitHub
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests. Anyone is welcome to submit issues, request features, or provide other feedback on GitHub. Please follow our Code of Conduct when making any requests.

## Please Reach Out
If you would like to directly contribute to the development of this project, please reach out to <mann@squared.ai> to get started and to learn our project's best practices for development! We thank you for your interest in helping support this project.

## Contributing Code

When contributing code to the BeyondML project, we ask that the following best practices be adhered to:

### Branching Structure

#### Main
The `main` branch is reserved for **production releases** of BeyondML. This branch is not expected to be committed directly to directly, except in extreme circumstances. Additionally, the `main` branch is expected to receive merges only from the `staging` branch.

#### Staging
The `staging` branch is reserved for **developmental/testing releases** of BeyondML. This branch is set up so that tests are run when code is committed to the branch, helping to ensure that all code that is part of any release is tested. The `staging` branch is expected to receive merges from "version branches."

#### Version Branches
"Version branches" are branches designed to create functionality to be released in the named version of the branch.  These branches are expected to be committed to directly by the core BeyondML team, and other developers are welcome to submit merge requests from their own personal branches.

### Testing
When new functionality is introduced, it is expected that tests be created to show that the functionality is working. Please update the files within the `./tests` directory with any tests you create. Additionally, it is very helpful if you check that all tests are passing before committing your code to the version branch. We recommend running the following command in your shell environment, from the top-level directory of the project, to test this:

```bash
pytest ./ -W ignore::DeprecationWarning
```

### Generating Documentation
We utilize [PDoc](https://pdoc.dev/) to generate the documentation for this project. When committing code to this project, specifically when code makes it to the `staging` branch, it is expected that the `./docs` folder be populated with up-to-date documentation for the package. To complete this, we recommend running the following command in your shell environment, from the top-level directory of the project, to generate the documentation:

```bash
pdoc -d numpy -o docs ./beyondml
```

We loosely follow [Numpy Documentation Conventions](https://numpydoc.readthedocs.io/en/latest/format.html) for this project, and the preceding line to generate the documentation will parse numpy documentation strings correctly.

Once again, thank you very much for your willingness and desire to contribute to the BeyondML project!
