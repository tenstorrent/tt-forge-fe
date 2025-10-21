[![Tests][tests badge]][tests]
[![Codecov][codecov badge]][codecov]
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tenstorrent/tt-forge-fe)

<div align="center">

<h1>

[Hardware](https://tenstorrent.com/cards/) | [Documentation](https://docs.tenstorrent.com/tt-forge-fe/) | [Discord](https://discord.gg/tenstorrent) | [Join Us](https://boards.greenhouse.io/tenstorrent?gh_src=22e462047us) | [Bounty $](https://github.com/tenstorrent/tt-forge-fe/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty)

</h1>

<img src="./docs/src/imgs/tt_refresh_forge-fe_w_logo_purple.png" alt="ttnn logo" height="230"/>

<br>

**TT-Forge FE** is a graph compiler designed to optimize and transform computational graphs for deep learning models, enhancing their performance and efficiency.

</div>

<br>

-----
# Quick Links
- [Getting Started / How to Run a Model](docs/src/getting_started.md)
- [Build](docs/src/getting_started_build_from_source.md) - Use these instructions if you plan to do development work.

-----
# What is this Repo?

TT-Forge-FE is a front end component within the broader tt-forge ecosystem, which is designed to compile and execute machine learning models on Tenstorrent hardware platforms like Wormhole and Blackhole. tt-forge-fe can ingest models from various machine learning frameworks including PyTorch, ONNX, and TensorFlow through the TVM Intermediate Representation (IR).

-----
# Current AI Framework Front End Projects
- [TT-Forge-FE](https://github.com/tenstorrent/tt-forge-fe)
  - A TVM based graph compiler designed to optimize and transform computational graphs for deep learning models. Supports ingestion of PyTorch, ONNX, TensorFlow, PaddlePaddle and similar ML frameworks via TVM ([TT-TVM](https://github.com/tenstorrent/tt-tvm)).
  - See [docs pages](https://docs.tenstorrent.com/tt-forge-fe/getting_started.html) for an overview and getting started guide.

- [TT-Torch](https://github.com/tenstorrent/tt-torch)

  - A MLIR-native, open-source, PyTorch 2.X and torch-mlir based front-end. It provides stableHLO (SHLO) graphs to TT-MLIR. Supports ingestion of PyTorch models via PT2.X compile and ONNX models via torch-mlir (ONNX->SHLO)
  - See [docs pages](https://docs.tenstorrent.com/tt-torch) for an overview and getting started guide.

- [TT-XLA](https://github.com/tenstorrent/tt-xla)
  - Leverages a PJRT interface to integrate JAX (and in the future other frameworks), TT-MLIR and Tenstorrent hardware. Supports ingestion of JAX models via jit compile, providing StableHLO (SHLO) graph to TT-MLIR compiler
  - See [Getting Started](docs/src/getting_started.md) for an overview and getting started guide.

-----
# Getting Started Guide

You can run a demo using the [TT-Forge-FE Getting Started](https://docs.tenstorrent.com/tt-forge-fe/getting-started.html) page.

-----
# Related Tenstorrent Projects
- [TT-Forge-FE](https://github.com/tenstorrent/tt-forge-fe)
- [TT-XLA](https://github.com/tenstorrent/tt-xla)
- [TT-Torch](https://github.com/tenstorrent/tt-torch)
- [TT-MLIR](https://github.com/tenstorrent/tt-mlir)
- [TT-Metalium](https://github.com/tenstorrent/tt-metal)
- [TT-TVM](https://github.com/tenstorrent/tt-tvm)


# Tenstorrent Bounty Program Terms and Conditions
This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-forge, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both “bounty” and difficulty level!
- - -

[codecov]: https://codecov.io/gh/tenstorrent/tt-forge-fe
[tests]: https://github.com/tenstorrent/tt-forge-fe/actions/workflows/on-push.yml?query=branch%3Amain
[deepwiki]: https://deepwiki.com/tenstorrent/tt-forge-fe
[codecov badge]: https://codecov.io/gh/tenstorrent/tt-forge-fe/graph/badge.svg
[tests badge]: https://github.com/tenstorrent/tt-forge-fe/actions/workflows/on-push.yml/badge.svg?query=branch%3Amain
[deepwiki badge]: https://deepwiki.com/badge.svg
