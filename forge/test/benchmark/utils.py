# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# Built-in modules

# Third-party modules
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor
from datasets import load_dataset
from ultralytics.nn.tasks import DetectionModel

# Forge modules


class YoloWrapper(torch.nn.Module):
    def __init__(self, url):
        super().__init__()
        self.model = self.load_model(url)
        self.model.model[-1].end2end = False  # Disable internal post processing steps

    def forward(self, image: torch.Tensor):
        y, x = self.model(image)
        # Post processing inside model casts output to float32, even though raw output is aligned with image.dtype
        # Therefore we need to cast it back to image.dtype
        return (y.to(image.dtype), *x)

    def load_model(self, url):
        # Load YOLO model weights
        weights = torch.hub.load_state_dict_from_url(url, map_location="cpu")

        # Initialize and load model
        model = DetectionModel(cfg=weights["model"].yaml)
        model.load_state_dict(weights["model"].float().state_dict())
        model.eval()

        return model


class Yolov4Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        x, y, z = self.model(image)
        # Post processing inside model casts output to float32,
        # even though raw output is aligned with image.dtype
        # Therefore we need to cast it back to image.dtype
        return x.to(image.dtype), y.to(image.dtype), z.to(image.dtype)


def load_benchmark_dataset(task, model_version, dataset_name, split, batch_size, loop_count):
    """
    Load the dataset for benchmarking.

    Parameters:
    ----------
    task: str
        The task to benchmark (e.g., "classification").
    model_version: str
        The version of the model to use.
    dataset_name: str
        The name of the dataset to load.
    split: str
        The split of the dataset to load (e.g., "train", "test").
    batch_size: int
        The batch size for the dataset.
    loop_count: int
        The number of times to loop through the dataset. Number of batches to process.

    Returns:
    -------
    inputs: list
        The input data for benchmarking.
    labels: list
        The labels for the input data.
    """

    if task == "classification":
        return load_dataset_classification(
            model_version=model_version,
            dataset_name=dataset_name,
            split=split,
            batch_size=batch_size,
            loop_count=loop_count,
        )
    else:
        raise ValueError(f"Unsupported task: {task}. Supported tasks are: classification.")


def load_dataset_classification(model_version, dataset_name, split, batch_size, loop_count):
    """
    Load the classification dataset for benchmarking.

    Parameters:
    ----------
    model_version: str
        The version of the model to use.
    dataset_name: str
        The name of the dataset to load.
    split: str
        The split of the dataset to load (e.g., "train", "test").
    batch_size: int
        The batch size for the dataset.
    loop_count: int
        The number of times to loop through the dataset. Number of batches to process.

    Returns:
    -------
    inputs: list
        The input data for benchmarking.
    labels: list
        The labels for the input data.
    """

    image_processor = AutoImageProcessor.from_pretrained(model_version)
    # Load the dataset as a generator
    dataset = iter(load_dataset(dataset_name, split=split, use_auth_token=True, streaming=True))
    inputs, labels = create_input_classification(dataset, image_processor, batch_size, loop_count)

    return inputs, labels


def create_batch_classification(dataset, image_processor, batch_size):
    """
    Create a batch of data for benchmarking.

    Parameters:
    ----------
    dataset: iterable
        The dataset to create the batch from.
    image_processor: AutoImageProcessor, ...
        The image processor to use for processing the images.
    batch_size: int
        The batch size for the dataset.

    Returns:
    -------
    X: torch.Tensor
        The input data for the batch.
    y: torch.Tensor
        The labels for the input data.
    """
    X, y = [], []
    # For each batch, we will get batch size number of samples
    for _ in tqdm(range(batch_size), desc="Creating the batch as number of samples"):
        # Get the next sample from the dataset, the next image and its label
        item = next(dataset)
        # Fetch the image and the label, decode them and add into the batch
        image = item["image"]
        label = item["label"]
        if image.mode == "L":
            image = image.convert(mode="RGB")
        temp = image_processor(image, return_tensors="pt")["pixel_values"]
        X.append(temp)
        y.append(label)
    X = torch.cat(X)
    y = torch.tensor(y)

    return X, y


def create_input_classification(dataset, image_processor, batch_size, loop_count):
    """
    Create input data for benchmarking. Input is made of the batches.

    Parameters:
    ----------
    dataset: iterable
        The dataset to create the batch from.
    image_processor: AutoImageProcessor, ...
        The image processor to use for processing the images.
    batch_size: int
        The batch size for the dataset.
    loop_count: int
        The number of times to loop through the dataset. Number of batches to process.

    Returns:
    -------
    inputs: list
        The input data for benchmarking.
    labels: list
        The labels for the input data.
    """

    inputs, labels = [], []
    # Number of batches we want to process is loop count
    for _ in tqdm(range(loop_count)):
        X, y = create_batch_classification(dataset, image_processor, batch_size)
        inputs.append(X)
        labels.append(y)

    return inputs, labels


def evaluate_classification(predictions, labels):
    """
    Evaluate the classification model.

    Parameters:
    ----------
    predictions: torch.Tensor
        The predictions made by the model.
    labels: torch.Tensor
        The true labels for the input data.

    Returns:
    -------
    target: float
        The accuracy of the model.
    """

    predictions = predictions.softmax(-1).argmax(-1)
    correct = (predictions == labels).sum()
    accuracy = 100.0 * correct / len(labels)
    accuracy = accuracy.item()

    return accuracy
