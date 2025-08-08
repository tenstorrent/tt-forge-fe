# Getting Started
This document walks you through how to set up TT-Forge-FE. TT-Forge-FE is a framework agnostic frontend that can convert any model to a generic Intermediate Representation (IR) that can then be converted to a Tenstorrent specific IR for use with Tenstorrent hardware. This is the main Getting Started page. There are two additional Getting Started pages depending on what you want to do. They are all described here, with links provided to each.

The following topics are covered:

* [Setup Options](#setup-options)
* [Configuring Hardware](#configuring-hardware)
* [Installing a Wheel and Running an Example](#installing-a-wheel-and-running-an-example)
* [Other Setup Options](#other-set-up-options)
    * [Using a Docker Container to Run an Example](getting_started_docker.md)
    * [Building From Source](getting_started_build_from_source.md)
* [Where to Go Next](#where-to-go-next)

> **NOTE:** If you encounter issues, please request assistance on the
>[TT-Forge-FE Issues](https://github.com/tenstorrent/tt-forge-fe/issues) page.

## Setup Options
TT-Forge-FE can be used to run models from any framework. Because TT-Forge-FE is open source, you can also develop and add features to it. Setup instructions differ based on the task. You have the following options, listed in order of difficulty:
* [Installing a Wheel and Running an Example](#installing-a-wheel-and-running-an-example) - You should choose this option if you want to run models.
* [Using a Docker Container to Run an Example](getting_started_docker.md) - Choose this option if you want to keep the environment for running models separate from your existing environment.
* [Building from Source](getting_started_build_from_source.md) - This option is best if you want to develop TT-Forge-FE further. It's a more complex process you are unlikely to need if you want to stick with running a model.

## Configuring Hardware
Before setup can happen, you must configure your hardware. You can skip this section if you already completed the configuration steps. Otherwise, this section of the walkthrough shows you how to do a quick setup using TT-Installer.

1. Configure your hardware with TT-Installer using the [Quick Installation section here.](https://docs.tenstorrent.com/getting-started/README.html#quick-installation)

2. Reboot your machine.

3. Make sure **hugepages** is enabled: 

```bash
sudo systemctl enable --now 'dev-hugepages\x2d1G.mount'
sudo systemctl enable --now tenstorrent-hugepages.service
```

4. Please ensure that after you run the TT-Installer script, after you complete reboot and set up hugepages, you activate the virtual environment it sets up - ```source ~/.tenstorrent-venv/bin/activate```.

5. After your environment is running, to check that everything is configured, type the following:

```bash
tt-smi
```

You should see the Tenstorrent System Management Interface. It allows you to view real-time stats, diagnostics, and health info about your Tenstorrent device.

![TT-SMI](./imgs/tt_smi.png)

## Installing a Wheel and Running an Example

This section walks you through downloading and installing a wheel. You can install the wheel wherever you would like if it is for running a model.

1. Make sure you are in an active virtual environment. This walkthrough uses the same environment you activated to look at TT-SMI in the [Configuring Hardware](#configuring-hardware) section.

2. For this walkthrough, TT-Forge-FE is used. You need to install two wheels for set up, **forge** and **tvm**. Install the wheels:

```bash
pip install tt_forge_fe --extra-index-url https://pypi.eng.aws.tenstorrent.com/
pip install tt_tvm  --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

3. Before you run a model, download and install the **MPI implementation**:

```bash
wget -q https://github.com/dmakoviichuk-tt/mpi-ulfm/releases/download/v5.0.7-ulfm/openmpi-ulfm_5.0.7-1_amd64.deb -O /tmp/openmpi-ulfm.deb && \
sudo apt install -y /tmp/openmpi-ulfm.deb
``` 

4. To test that everything is running correctly, try an example model. You can use nano or another text editor to paste this code into a file named **forge_example.py** and then run it from the terminal. You should still have your virtual environment running after installing the wheels when running this example:

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

5. You have now set up the latest wheels for TT-Forge-FE, and can run any models you want inside your virtual environment.

## Other Set up Options
If you want to keep your environment completely separate in a Docker container, or you want to develop TT-Forge-FE further, this section links you to the pages with those options:

* [Setting up a Docker Container](getting_started_docker.md) - keep everything for running models in a container
* [Building from Source](getting_started_build_from_source.md) - set up so you can develop TT-Forge-FE

## Where to Go Next

Now that you have set up TT-Forge-FE, you can compile and run other demos or your own models. See the [TT-Forge-FE folder in the TT-Forge repo](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-forge-fe) for more demo options.
