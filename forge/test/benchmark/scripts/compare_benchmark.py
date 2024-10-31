# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# =============================================================================
# Import the required libraries
# =============================================================================

import argparse
import os


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
    repo_url: str
        The name of the repository to access.

    token: str
        The github token to access the repository.

    commit: str
        The commit hash that represent change for which we want to check results.

    job: str
        The job id that represent the job for which we want to check results.

    Attributes:
    -----------
    repo_url: str
        The name of the repository to access.

    token: str
        The github token to access the repository.

    commit: list[str]
        The commit hash that represent change for which we want to check results.

    job: list[str]
        The job id that represent the job for which we want to check results.

    Exceptions:
    -----------
    None

    Examples:
    ---------
    None

    """

    def __init__(self, repo_url: str, token: str, commit: list[str] = None, job: list[str] = None):

        # Initialize the attributes
        # All the attributes are private, also,
        # repository name cannot be changed once set through the constructor.

        self.__repo_url = repo_url
        self.__token = token
        self.__commit = commit
        self.__job = job

    @property
    def repo_url(self):
        return self.__repo_url

    @property
    def token(self):
        return self.__token

    @token.setter
    def token(self, value):
        self.__token = value

    @property
    def commit(self):
        return self.__commit

    @commit.setter
    def commit(self, value):
        self.__commit = value

    @property
    def job(self):
        return self.__job

    @job.setter
    def job(self, value):
        self.__job = value

    def __str__(self):

        string = f"GithubManager\n"
        string += "====================\n"
        string += f"repo_url: {self.repo_url}\n"
        string += f"token: {self.token}\n"
        if self.commit:
            string += f"commit={self.commit}\n"
        if self.job:
            string += f"job={self.job}\n"
        string += "====================\n"

        return string


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


class BenchmarkManager:

    """
    Class to manipulate the benchmark results.

    Parameters:
    -----------
    github_manager: GithubManager
        The github manager to access the repository.

    Attributes:
    -----------
    github_manager: GithubManager
        The github manager to access the repository.

    benchmarks: list[Benchmark]
        The list of objects of the Benchmark class. Benchmark class represents the benchmark results.

    Exceptions:
    -----------

    Examples:
    ---------

    """

    def __init__(self, github_manager: GithubManager):
        self.__github_manager = github_manager
        self.__benchmarks = []

    @property
    def github_manager(self):
        return self.__github_manager

    @property
    def benchmarks(self):
        return self.__benchmarks

    def add(self, benchmark: Benchmark):
        self.__benchmarks.append(benchmark)

    def __add__(self, benchmark: Benchmark):
        self.add(benchmark)

    def read_commits():
        """Read the commits from the github repository."""
        pass

    def read_jobs():
        """Read the jobs from the github repository."""
        pass

    def extract():
        pass

    def compare():
        pass

    def print_results():
        pass

    def __str__(self):
        """
        Print the BenchmarkManager object.
        It includes the github manager and the benchmarks.
        Benchmark has its own __str__ method.
        """

        string = f"BenchmarkManager\n"
        string += "=" * 20 + "\n"
        string += f"github_manager: {self.github_manager}\n"
        string += f"benchmarks: \n"
        string += "-" * 20 + "\n"
        for benchmark in self.benchmarks:
            string += f"{benchmark}\n"
        string += "=" * 20 + "\n"

        return string


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
    ValueError
        If the commits or jobs are empty.

    Examples:
    ---------

    """

    # Initialize the parsed arguments
    parsed_args = {}

    # Read the arguments from the command line
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
    token: str
        The github token.

    Exceptions:
    -----------
    ValueError
        If the github token is not set in the environment variable

    Examples:
    ---------

    """

    # Extract the github token from the environment variable.
    token = os.getenv("GITHUB_TOKEN")
    if not token:

        # Try to read the token from the file.
        try:
            with open("github_token.txt", "r") as file:
                token = file.read().strip()
        except FileNotFoundError as e:
            print("\nThe github token is not set in the environment variable GITHUB_TOKEN.\n")
            raise ValueError("The github token is not set in the environment variable GITHUB_TOKEN")

    else:
        with open("github_token.txt", "w") as file:
            file.write(token)

    return token


def main():
    """
    Main function for comparing the benchmark results.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """

    # Read github token
    token = read_token()

    # Read the arguments from the command line.
    parsed_args = read_args()

    # Create the GithubManager.
    github_manager = GithubManager(repo_url=GITHUB_REPO_URL_TT_FORGE_FE, token=token)
    if parsed_args.get("commits"):
        github_manager.commit = parsed_args["commits"]
    if parsed_args.get("jobs"):
        github_manager.job = parsed_args["jobs"]

    print(github_manager)

    # Create the BenchmarkManager.
    benchmark_manager = BenchmarkManager(github_manager=github_manager)

    # Create the Benchmark.

    # Process and compare the benchmark results.

    # Print the results.

    print("\nBenchmark comparing done.\n")


if __name__ == "__main__":
    main()
