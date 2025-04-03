# Contributing to dattri

We welcome contributions from the community and are pleased to have you join us. This document is intended to help you understand the process for contributing to the project, setting up your development environment, and ensuring that your contributions adhere to our coding standards.

## Setting Up Your Development Environment

Before you can contribute to the project, you need to set up your development environment. Here are the steps to do so:

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/TRAIS-Lab/dattri.git
cd dattri
```

### 2. Install Dependencies

Install the necessary dependencies:

```bash
pip install -e .
```

#### Recommended enviroment setup
It's **not** required to follow the exact same steps in this section. But this is a verified environment setup flow that may help users to avoid most of the issues during the installation.

```bash
conda create -n dattri python=3.10
conda activate dattri

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

pip install -e .
```

### 3. Install Pre-commit Hooks

We use pre-commit hooks to ensure that your contributions adhere to our code standards. To set up pre-commit hooks in your local repository, follow these steps:

```bash
pip install pre-commit
pre-commit install
```

## Making Changes
When you're ready to make changes, please follow these steps:

### 1. Create a Branch

Create a new branch for your changes:

```bash
git checkout -b your-feature-branch
```

### 2. Make Your Changes

Implement your feature or fix bugs and make sure that your changes are well documented.

### 3. Run Tests

Run `pytest test/<your test file>` on the test files you implemented (if applicable).

### 4. Run Pre-commit Hooks

Before committing your changes, pre-commit hooks will automatically run when you attempt to make a commit with the following command:

```bash
git commit -m "Add a detailed commit message"
```

You will see something like this:
```
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Passed
Check Yaml...........................................(no files to check)Skipped
Check for added large files..............................................Passed
ruff.................................................(no files to check)Skipped
ruff-format..........................................(no files to check)Skipped
[fix-pr-template 9323f57] fix pr template
 1 file changed, 1 insertion(+), 1 deletion(-)
```

Make sure all hooks pass. If any hooks fail, you will need to address the failures and try committing again.
