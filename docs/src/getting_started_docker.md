# Getting Started with Docker and Tenstorrent
This document walks you through how to set up TT-Forge-FE using a Docker image. There are two other available options for getting started: 
* [Installing a Wheel](getting_started.md) - if you do not want to use Docker, and prefer to use a virtual environment by itself instead, use this method.
* [Building from Source](getting_started_build_from_source.md) - if you plan to develop TT-Forge-FE further, you must build from source, and should use this method. 

## Configuring Hardware
Before setup can happen, you must configure your hardware. This section of the walkthrough shows you how to do a quick setup with Tenstorrent's TT-Installer. 

1. Configure your hardware with TT-Installer:

```bash
/bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"
```

> **NOTE:** This walkthrough assumes that you use the [Quick Installation]
> (https://docs.tenstorrent.com/getting-started/README.html#quick-installation) instructions for set up.

2. Reboot your machine. 

3. Please ensure that after you run this script, after you complete reboot, you activate the virtual environment it sets up - ```source ~/.tenstorrent-venv/bin/activate```.

4. When your environment is running, to check that everything is configured, type the following: 

```bash
tt-smi
```

You should see the Tenstorrent System Management Interface. It allows you to view real-time stats, diagnostics, and health info about your Tenstorrent device. 

![TT-SMI](./imgs/tt_smi.png)

5. You can now deactivate the virtual environment.

## Setting up the Docker Container

This section walks through the installation steps for using a Docker container for your project.

To install, do the following:

1. Install Docker if you do not already have it:

```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

2. Test that Docker is installed:

```bash
docker --version
```

3. Add your user to the Docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

4. Run the Docker container:

```bash
docker run -it --rm \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  ghcr.io/tenstorrent/tt-forge/tt-forge-slim:dev20250618
```

5. If you want to check that it is running, open a new tab with the **Same Command** option and run the following:

```bash
docker ps
```

6. You are now ready to move on to the next section, and run your first model. 

## Running Models in Docker
This section shows you how to run a model using Docker. The provided example is from the TT-Forge repo. Do the following:

1. Inside your running Docker container, clone the TT-Forge repo:

```bash
git clone https://github.com/tenstorrent/tt-forge.git
```

2. Activate the virtual environment provided for TT-Forge:

```bash
source venv-tt-forge-fe/bin/activate
```

3. Run a model. For this set up, the **mobile_netv2_demo.py** is used:

```bash
python tt-forge/demos/tt-forge-fe/cnn/mobile_netv2_demo.py
```

4. If all goes well you will get the following output: 

Prediction: Samoyed, Samoyede (class 258)
Confidence: 0.868

## Where to Go Next

Now that you have set up TT-Forge-FE, you can compile and run your own models.

For a quick start, here is an example of a custom model. Note the introduction of the `forge.compile` call:

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
