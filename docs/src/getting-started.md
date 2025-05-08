# Getting Started
This document walks you through how to set up to run models using tt-forge. The following topics are covered:

* [Configuring Hardware](#configuring-hardware)
* [Setting up the Docker Container](#setting-up-the-docker-container)
* [Installing Dependencies](#installing-depencencies)
* [Creating a Virtual Environment](#creating-a-virtual-environment)
* [Installing a Wheel](#installing-a-wheel)
* [Running a Demo](#running-a-demo)


> **NOTE:** If you encounter issues, please request assistance on the
>[tt-forge-fe Issues](https://github.com/tenstorrent/tt-forge-fe/issues) page.

> **NOTE:** If you plan to do development work in the tt-forge repo, you need to build from source. To do that, please see the
> [build instructions for tt-forge-fe](https://github.com/tenstorrent/tt-forge-fe/
> blob/main/docs/src/build.md).

## Configuring Hardware 

Configure your Tenstorrent hardware before continuing. Use the [Starting Guide](https://docs.tenstorrent.com/getting-started/README.html).

> **NOTE:** This walkthrough assumes that you use the [Quick Installation]
> (https://docs.tenstorrent.com/getting-started/README. html#quick-installation) instructions for set up. Please ensure that after
> you run this script, you activate the virtual environment it sets up - ```source ~/.tenstorrent-venv/bin/activate```.

## Setting up the Docker Container

The simplest way to run models is to use one of the Docker images. There are two Docker images you can use to set up your environment:

* **Base Image**: This image includes all the necessary dependencies.
    * ghcr.io/tenstorrent/tt-forge-fe/tt-forge-fe-base-ird-ubuntu-22-04
* **Prebuilt Environment Image**: This image contains all necessary dependencies and a prebuilt environment.
    * ghcr.io/tenstorrent/tt-forge-fe/tt-forge-fe-ird-ubuntu-22-04

To install, do the following:

1. Install Docker if you do not already have it:

```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

2. Test that docker is installed:

```bash
docker --version
```

3. Add your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

4. Run the container (the prebuilt image is used here):

```bash
docker run -it ghcr.io/tenstorrent/tt-forge-fe/tt-forge-fe-ird-ubuntu-22-04
```

5. If you want to check that it's running, open a new tab with the **Same Command** option and run the following:

```bash
docker ps
```

## Creating a Virtual Environment
It is recommended that you install a virtual environment for the wheel you want to work with. Wheels from different repos may have conflicting dependencies.

Create a virtual environment:

```bash
python3 -m venv nameofenvironment-venv
source nameofenvironment/bin/activate
```

## Installing a Wheel 
This section walks you through downloading and installing a wheel. You can install the wheel wherever you would like if it's for running a model. If you want to do development work, you must clone the repo you want, navigate into it, and then set up the wheel.

1. Make sure you are in an active virtual environment.

> **NOTE**: If you plan to do development work, before continuing with these instructions, clone the repo you plan to use, then navigate into the repo. If you are just running models, this step is not necessary.

2. Download the wheel(s) you want to use from the [Tenstorrent Nightly Releases](https://github.com/tenstorrent/tt-forge/releases) page.

For this walkthrough, tt-forge-fe is used. You need to install two wheels for set up:

```bash
pip install https://github.com/tenstorrent/tt-forge/releases/download/0.1.0.dev20250422214451/forge-0.1.0.dev20250422214451-cp310-cp310-linux_x86_64.whl
```

```bash
pip install https://github.com/tenstorrent/tt-forge/releases/download/0.1.0.dev20250422214451/tvm-0.1.0.dev20250422214451-cp310-cp310-linux_x86_64.whl
```

> **NOTE:** The commands are examples, for the latest install link, go to the
> [Tenstorrent Nightly Releases](https://github.com/tenstorrent/tt-forge/releases)
> page. The generic download will be:
> `https://github.com/tenstorrent/tt-forge/releases/download/0.1.0.devDATE/
> NAMEOFWHEEL`
>
> If you plan to work with wheels from different repositories, make a separate
> environment for each one. Some wheels have conflicting dependencies.

## Run First Example Case

To confirm that our environment is properly setup, let's run one sanity test for element-wise add operation:
```bash
pytest forge/test/mlir/operators/eltwise_binary/test_eltwise_binary.py::test_add
```

In a few seconds, you should get confirmation if this test passed successfully. Once that's done, we can run one of our model tests as well:
```bash
pytest forge/test/mlir/llama/tests/test_llama_prefil.py::test_llama_prefil_on_device_decode_on_cpu
```

## Running Models 

You can try one of the models in the tt-forge repo. For a list of models that work with tt-forge-fe, navigate to the [Demos](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-forge-fe) folder in the tt-forge repo. Follow the [Getting Started](https://github.com/tenstorrent/tt-forge/blob/main/docs/src/getting-started.md) instructions there. 

## Where to Go Next

Now that you have set up tt-forge-fe, you can compile and run your own models.

For a quick start, here is an example of how to run your own model. Note the introduction of the `forge.compile` call:

```py
import torch
from transformers import ResNetForImageClassification

def resnet():
    # Load image, pre-process, etc.
    ...

    # Load model (e.g. from HuggingFace)
    framework_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    # Compile the model using Forge
    compiled_model = forge.compile(framework_model, input_image)

    # Run compiled model
    logits = compiled_model(input_image)

    ...
    # Post-process output, return results, etc.
```
