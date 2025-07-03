# Getting Started with Docker
This document walks you through how to set up TT-Forge-FE using a Docker image. There are two other available options for getting started:
* [Installing a Wheel](getting_started.md) - if you do not want to use Docker, and prefer to use a virtual environment by itself instead, use this method.
* [Building from Source](getting_started_build_from_source.md) - if you plan to develop TT-Forge-FE further, you must build from source, and should use this method.

The following topics are covered:

* [Configuring Hardware](#configuring-hardware)
* [Setting up the Docker Container](#setting-up-the-docker-container)
* [Running Models in Docker](#running-models-in-docker)
* [Where to Go Next](#where-to-go-next)

> **NOTE:** If you encounter issues, please request assistance on the
>[TT-Forge-FE Issues](https://github.com/tenstorrent/tt-forge-fe/issues) page.

## Configuring Hardware
Before setup can happen, you must configure your hardware. You can skip this section if you already completed the configuration steps. Otherwise, this section of the walkthrough shows you how to do a quick setup using TT-Installer.

1. Configure your hardware with TT-Installer using the [Quick Installation section here.](https://docs.tenstorrent.com/getting-started/README.html#quick-installation)

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
  ghcr.io/tenstorrent/tt-forge/tt-forge-fe-slim:latest
```

5. If you want to check that it is running, open a new tab with the **Same Command** option and run the following:

```bash
docker ps
```

6. To check that everything is running as expected, try an example model. You can use nano or another text editor to paste this code into a file named **forge_example.py** and then run it from the terminal:

```python
import torch
import forge

class Add(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, a, b):
    return a + b

a = torch.rand(size=(2, 32, 32))
b = torch.rand(size=(2, 32, 32))

framework_module = Add()
compiled_model = forge.compile(framework_module, sample_inputs=[a, b])

out = compiled_model(a, b)

print("compiled output:", out)
```

7. If all goes well, you are now ready to move on to the next section, and run your first demo model.

## Running Models in Docker
This section shows you how to run a model using Docker. The provided example is from the TT-Forge repo. Do the following:

1. Inside your running Docker container, clone the TT-Forge repo:

```bash
git clone https://github.com/tenstorrent/tt-forge.git
```

2. Set the path for Python:

```bash
export PYTHONPATH=/tt-forge:$PYTHONPATH
```

3. Navigate into TT-Forge and run the following command:

```bash
git submodule update --init --recursive
```

4. Navigate back out of the TT-Forge directory.

5. Run a model. For this set up, the **mobile_netv2_demo.py** is used:

```bash
python tt-forge/demos/tt-forge-fe/cnn/mobile_netv2_demo.py
```

6. If all goes well you will get the following output:

Prediction: Samoyed, Samoyede (class 258)
Confidence: 0.868

## Where to Go Next

Now that you have set up TT-Forge-FE, you can compile and run your own models. See the [TT-Forge-FE folder in the TT-Forge repo](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-forge-fe) for more demo options.

For a quick start creating your own model, here is an example of a custom model. Note the introduction of the `forge.compile` call:

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
