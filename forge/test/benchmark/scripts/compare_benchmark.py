# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# =============================================================================
# Import the required libraries
# =============================================================================

import argparse


# =============================================================================
# Constants
# =============================================================================

GITHUB_REPO_URL_TT_FORGE_FE = "https://github.com/tenstorrent/tt-forge-fe"


# =============================================================================
# Define the data structures
# =============================================================================


class GithubManager:

    """
    Class to manage the github repository and its contents.

    Parameters:
    -----------
    repo_name: str
        The name of the repository to access.

    token: str
        The github token to access the repository.

    commit: str
        The commit hash that represent change for which we want to check results.

    job: str
        The job id that represent the job for which we want to check results.

    Attributes:
    -----------
    repo_name: str
        The name of the repository to access.

    token: str
        The github token to access the repository.

    commit: str
        The commit hash that represent change for which we want to check results.

    job: str
        The job id that represent the job for which we want to check results.

    Exceptions:
    -----------
    None

    Examples:
    ---------
    None

    """

    def __init__(self, repo_name: str, token: str, commit: str = None, job: str = None):
        self.repo_name = repo_name
        self.token = token
        self.commit = commit
        self.job = job

    @property
    def repo_name(self):
        pass

    @property
    def token(self):
        pass

    @token.setter
    def token(self, value):
        pass

    @property
    def commit(self):
        pass

    @commit.setter
    def commit(self, value):
        pass

    @property
    def job(self):
        pass

    @job.setter
    def job(self, value):
        pass

    def __str__(self):

        string = f"GithubManager\n"
        string += "====================\n"
        string += f"token: {self.token}\n"
        if self.commit:
            string += f"commit={self.commit}\n"
        if self.job:
            string += f"job={self.job}\n"
        string += "====================\n"

        return string


class BenchmarkManager:

    """
    Class to manipulate the benchmark results.

    Parameters:
    -----------

    Attributes:
    -----------

    Exceptions:
    -----------

    Examples:
    ---------

    """

    def __init__(self):
        pass

    def read():
        pass

    def extract():
        pass

    def __str__(self):
        return "BenchmarkManager"


class Benchmark:

    """
    Class to represent the benchmark results.

    Parameters:
    -----------
    results: dict
        The results of the benchmark. The dictionary contains all information about the benchmark model,
        the machine on which the benchmark was run, and the results of the benchmark.

    Attributes:
    -----------
    results: dict
        The results of the benchmark. The dictionary contains all information about the benchmark model,
        the machine on which the benchmark was run, and the results of the benchmark.

    model: str
        The name of the model for which the benchmark was run.

    Exceptions:
    -----------
    ValueError
        If the benchmark results are empty.

    Examples:
    ---------

    """

    def __init__(self, results: dict):
        self.results = results
        if not self.results:
            raise ValueError("The benchmark results are empty.")
        self.model = self.results["model"]

    def compare():
        pass

    def __str__(self):
        return f"Benchmark: {self.model}"


# =============================================================================
# Define the functions
# =============================================================================


def read_args():
    """
    Read the arguments from the command line.

    Parameters:
    -----------
    None

    Returns:
    --------
    parsed_args: dict
        The parsed arguments from the command line.

    Exceptions:
    -----------

    Examples:
    ---------

    """

    parser = argparse.ArgumentParser(description="Compare the benchmark results.")
    parser.add_argument(
        "-c",
        "--commits",
        type=str,
        help="The commit hash to compare the benchmark results. Pass the commits as a comma separated list.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=str,
        help="The job id to compare the benchmark results. Pass the jobs as a comma separated list.",
    )

    args = parser.parse_args()

    # Initialize the parsed arguments
    parsed_args = {}

    if args.commits and args.jobs:
        print("\nPass only commits or jobs.\n\n")
        print(parser.print_help())
        exit(1)

    if not args.commits and not args.jobs:
        print("\nEither commits or jobs must be specified.\n\n")
        print(parser.print_help())
        exit(1)

    if args.commits:
        commits = args.commits.split(",")
        if not commits:
            raise ValueError(
                "The commits are empty. They shpuld be a comma separated list. For example: fh3g2f3h, 3h4h3h4"
            )
        parsed_args["commits"] = commits

    if args.jobs:
        jobs = args.jobs.split(",")
        if not jobs:
            raise ValueError("The jobs are empty. They shpuld be a comma separated list. For example: 1234, 5678")
        parsed_args["jobs"] = jobs

    return parsed_args


def read_token():
    """
    Read the github token from the environment variable.

    Parameters:
    -----------

    Returns:
    --------

    Exceptions:
    -----------

    Examples:
    ---------

    """
    pass


def read_commit():
    """
    Read the commit hash from the command line.

    Parameters:
    -----------

    Returns:
    --------

    Exceptions:
    -----------

    Examples:
    ---------

    """
    pass


def read_job():
    """
    Read the job id from the command line.

    Parameters:
    -----------

    Returns:
    --------

    Exceptions:
    -----------

    Examples:
    ---------

    """
    pass


def main():

    # Read the arguments from the command line.

    # Create the GithubManager.

    # Create the BenchmarkManager.

    # Create the Benchmark.

    # Process and compare the benchmark results.

    # Print the results.

    print("\nBenchmark comparing done.\n")


if __name__ == "__main__":
    main()
