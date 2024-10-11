# Tools

This section will cover setup of various tools that can help you with development of tt-forge-fe.

## Pre-commit

We have defined various pre-commit hooks that check the code for formatting, licensing issues, etc.

To install pre-commit, run the following command:

```sh
source env/activate
pip install pre-commit
```

After installing pre-commit, you can install the hooks by running:

```sh
pre-commit install
```

Now, each time you run `git commit` the pre-commit hooks (checks) will be executed.

If you have already committed before installing the pre-commit hooks, you can run on all files to "catch up":

```sh
pre-commit run --all-files
```

For more information visit [pre-commit](https://pre-commit.com/)

## mdbook

We use `mdbook` to generate the documentation. To install `mdbook` on Ubuntu, run the following commands:

```sh
sudo apt install cargo
cargo install mdbook
```

>**NOTE:** If you don't want to install `mdbook` via cargo (Rust package manager), or this doesn't work for you, consult the [official mdbook installation guide](https://rust-lang.github.io/mdBook/cli/index.html).
