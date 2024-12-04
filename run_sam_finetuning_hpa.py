"""Finetuning Segment Anything using 'µsam'.

This python script shows how to use Segment Anything for Microscopy to fine-tune a Segment Anything Model (SAM)
on open-source microscopy images with multiple channels.

We use confocal microscopy images from the HPA Kaggle Challenge for protein identification
(from Ouyang et al. - https://doi.org/10.1038/s41592-019-0658-6) in this script for the cell segmentation task.
The functionalities shown here should work for your (microscopy) images too.
"""

import os
from typing import Union, Tuple, Literal, Optional, List

import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components

import torch
from torch.utils.data import DataLoader

from torch_em.util.debug import check_loader
from torch_em.util.util import get_random_colors

from micro_sam_ray import util
import micro_sam_ray.training as sam_training
from micro_sam_ray.training.util import normalize_to_8bit
from micro_sam_ray.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from download_datasets import _get_hpa_data_paths

import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def download_dataset(
    path: Union[os.PathLike, str], split: Literal['train', 'val', 'test'], download: bool = True,
) -> Tuple[List[str], List[str]]:
    """Download the HPA dataset.

    This functionality downloads the images and corresponding labels stored as `tif` files.

    Args:
        path: Filepath to the directory where the data will be stored.
        split: The choice of data split. Either 'train', 'val' or 'test'.
        download: Whether to download the dataset.

    Returns:
        List of filepaths for the image data.
        List of filepaths for the label data.
    """
    image_paths, label_paths = _get_hpa_data_paths(path=path, split=split, download=download, downloader="owncloud")
    return image_paths, label_paths


def verify_inputs(image_paths: List[str], label_paths: List[str]):
    """Verify the downloaded inputs and understand the inputs' data structure.

    Args:
        image_paths: List of filepaths for the image data.
        label_paths: List of filepaths for the label data.
    """
    for image_path, label_path in zip(image_paths, label_paths):
        image = imageio.imread(image_path)
        labels = imageio.imread(label_path)

        # The images should be of shape: H, W, 4 -> where, 4 is the number of channels.
        if (image.ndim == 3 and image.shape[-1] == 3) or image.ndim == 2:
            print(f"Inputs '{image.shape}' match the channel expectations.")
        else:
            print(f"Inputs '{image.shape}' must match the channel expectations (of either one or three channels).")

        # The labels should be of shape: H, W
        print(f"Shape of corresponding labels: '{labels.shape}'")

        break  # comment this line out in case you would like to verify the shapes for all inputs.


def preprocess_inputs(image_paths: List[str]):
    """Preprocess the input images.

    NOTE: For this part, we choose to keep `microtubules`, `proten` and `nuclei` channels as the first,
    second and third channels respectively for finetuning Segment Anything.

    Args:
        image_paths: List of filepaths for the image data.
    """
    # We remove the 'er' channel, i.e. the last channel.
    for image_path in image_paths:
        image = imageio.imread(image_path)

        if image.ndim == 3 and image.shape[-1] == 4:  # Convert 4 channel inputs to 3 channels.
            image = image[..., :-1]
            imageio.imwrite(image_path, image)


def visualize_inputs(image_paths: List[str], label_paths: List[str]):
    """Visualize the images and corresponding labels.

    Args:
        image_paths: List of filepaths for the image data.
        label_paths: List of filepaths for the label data.
    """
    for image_path, label_path in zip(image_paths, label_paths):
        image = imageio.imread(image_path)
        labels = imageio.imread(label_path)

        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input Image")
        ax[0].axis("off")

        labels = connected_components(labels)
        ax[1].imshow(labels, cmap=get_random_colors(labels), interpolation="nearest")
        ax[1].set_title("Ground Truth Instances")
        ax[1].axis("off")

        plt.show()
        plt.close()

        break  # comment this out in case you want to visualize all the images


def get_dataloaders(
    train_image_paths: List[str],
    train_label_paths: List[str],
    val_image_paths: List[str],
    val_label_paths: List[str],
    view: bool,
    train_instance_segmentation: bool,
) -> Tuple[DataLoader, DataLoader]:
    """Get the HPA dataloaders for cell segmentation.

    NOTE: `micro_sam.training.default_sam_loader` is a convenience function to build a torch dataloader
    from the image data and labels for training segmentation models.
    This is wrapped around the `torch_em.default_segmentation_loader`.
    It suppoirts image data in various formats. Here, we load image data and corresponding labels by providing
    filepaths to the respective tif files that were downloaded and preprocessed using the functionality above.

    For more information, here is the documentation:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/README.md

    Here is a detailed notebook on finetuning Segment Anything:
    https://github.com/computational-cell-analytics/micro-sam/blob/master/notebooks/sam_finetuning.ipynb

    Args:
        train_image_paths: List of filepaths for the training image data.
        train_label_paths: List of filepaths for the training label data.
        val_image_paths: List of filepaths for the validation image data.
        val_label_paths: List of filepaths for the validation label data.
        view: Whether to view the samples out of training dataloader.
        train_instance_segmentation: Whether to finetune SAM with additional instance segmentation decoder.

    Returns:
        The PyTorch DataLoader for training.
        The PyTorch DataLoader for validation.
    """

    train_image_paths = [os.path.abspath(path) for path in train_image_paths]
    train_label_paths = [os.path.abspath(path) for path in train_label_paths]
    val_image_paths = [os.path.abspath(path) for path in val_image_paths]
    val_label_paths = [os.path.abspath(path) for path in val_label_paths]

    # Debug: Print paths
    # print("Train image paths:", train_image_paths)
    # print("Train label paths:", train_label_paths)
    
    # Load images from tif stacks by setting `raw_key` and `label_key` to None.
    raw_key, label_key = None, None

    batch_size = 1  # the training batch size
    patch_shape = (1024, 1024)  # the size of patches for training

    # The dataloader internally takes care of adding label transforms: i.e. used to convert the ground-truth
    # labels to the desired instances for finetuning Segment Anythhing, or, to learn the foreground and distances
    # to the object centers and object boundaries for automatic segmentation.

    train_loader = sam_training.default_sam_loader_distributed(
        raw_paths=train_image_paths,
        raw_key=raw_key,
        label_paths=train_label_paths,
        label_key=label_key,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        with_channels=True,
        with_segmentation_decoder=train_instance_segmentation,
        batch_size=batch_size,
        shuffle=True,
        raw_transform=normalize_to_8bit,
        n_samples=100,
    )

    val_loader = sam_training.default_sam_loader_distributed(
        raw_paths=val_image_paths,
        raw_key=raw_key,
        label_paths=val_label_paths,
        label_key=label_key,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        with_channels=True,
        with_segmentation_decoder=train_instance_segmentation,
        batch_size=batch_size,
        shuffle=True,
        raw_transform=normalize_to_8bit,
    )

    if view:
        # Let's check how our samples look from the dataloader.
        check_loader(train_loader, 4, plt=True)

    return train_loader, val_loader


def run_finetuning(
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_root: Optional[Union[os.PathLike, str]],
    train_instance_segmentation: bool,
    device: Union[torch.device, str],
    model_type: str,
    overwrite: bool,
) -> str:
    """Run finetuning for the Segment Anything model on microscopy images using Ray for distributed training."""
    # All hyperparameters for training
    n_objects_per_batch = 5
    n_epochs = 20
    checkpoint_name = "sam_hpa"

    if save_root is None:
        save_root = os.getcwd()

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    # Create the train configuration
    train_config = {
        "name": checkpoint_name,
        "save_root": save_root,
        "model_type": model_type,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "n_epochs": n_epochs,
        "lr": 1e-5,
        "wd": 1e-2,
        "n_objects_per_batch": n_objects_per_batch,
        "with_segmentation_decoder": train_instance_segmentation,
        "device": "ray",
    }

    # Configure the scaling for distributed training
    scaling_config = ScalingConfig(
        num_workers=2,  # Adjust according to your available resources
        use_gpu=True,
        resources_per_worker={
            "CPU": 4,
            "GPU": 1,
        },
    )

    # Set up the trainer using Ray's TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=sam_training.train_sam_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=ray.train.RunConfig(
            storage_path=os.path.join(save_root, "tmp", "sam_finetuning_ray"),
            name="sam_finetuning_ray",
        ),
    )

    # Run distributed training
    result = trainer.fit()

    # Retrieve the actual checkpoint directory
    checkpoint_dir = result.checkpoint.to_directory()  # Extracts the checkpoint to a local directory
    checkpoint_path = os.path.join(checkpoint_dir, "sam_hpa", "best.pt")

    # Verify the checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Please check the training process or the checkpoint directory."
        )

    return checkpoint_path


import os

def run_instance_segmentation_with_decoder(
    test_image_paths: List[str], model_type: str, checkpoint: Union[os.PathLike, str], device: Union[torch.device, str],
):
    """Run automatic instance segmentation (AIS) and save the results to the 'ray_tuning_result' folder.

    Args:
        test_image_paths: List of filepaths for the test image data.
        model_type: The choice of Segment Anything model (connotated by the size of image encoder).
        checkpoint: Filepath to the finetuned model checkpoints.
        device: The torch device used for inference.
    """
    assert os.path.exists(checkpoint), "Please train the model first to run inference on the finetuned model."

    # Create the output directory
    output_dir = "tmp/ray_tuning_result"
    os.makedirs(output_dir, exist_ok=True)

    # Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, checkpoint=checkpoint, device=device, is_tiled=True
    )

    for idx, image_path in enumerate(test_image_paths):
        image = imageio.imread(image_path)
        image = normalize_to_8bit(image)

        # Predicting the instances.
        prediction = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter, input_path=image, ndim=2, tile_shape=(768, 768), halo=(128, 128)
        )

        # Save the input image and prediction as separate files
        input_image_path = os.path.join(output_dir, f"input_image_{idx}.png")
        prediction_image_path = os.path.join(output_dir, f"prediction_{idx}.png")

        # Save the input image
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title("Input Image")
        plt.savefig(input_image_path, bbox_inches="tight")
        plt.close()

        # Save the prediction
        plt.figure(figsize=(10, 10))
        plt.imshow(prediction, cmap=get_random_colors(prediction), interpolation="nearest")
        plt.axis("off")
        plt.title("Predictions (AIS)")
        plt.savefig(prediction_image_path, bbox_inches="tight")
        plt.close()

        print(f"Saved input image to: {input_image_path}")
        print(f"Saved prediction to: {prediction_image_path}")

        # Uncomment the following line if you want to process all images
        # break  # Comment this out to process all images


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run finetuning for Segment Anything model for microscopy images.")
    parser.add_argument(
        "-i", "--input_path", type=str, default="./data",
        help="The filepath to the folder where the image data will be downloaded. "
        "By default, the data will be stored in your current working directory at './data'."
    )
    parser.add_argument(
        "-s", "--save_root", type=str, default=None,
        help="The filepath to store the model checkpoint and tensorboard logs. "
        "By default, they will be stored in your current working directory at 'checkpoints' and 'logs'."
    )
    parser.add_argument(
        "--view", action="store_true",
        help="Whether to visualize the raw inputs, samples from the dataloader, instance segmentation outputs, etc."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to overwrite the already finetuned model checkpoints."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="The choice of device to run training and inference."
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="The number of worker processes for distributed training.",
    )
    args = parser.parse_args()

    device = 'ray'  # Indicate that we are using Ray for distributed training

    # The model_type determines which base model is used to initialize the weights that are finetuned.
    model_type = "vit_b"

    # Train an additional convolutional decoder for end-to-end automatic instance segmentation
    train_instance_segmentation = True

    # Step 1: Download the dataset.
    train_image_paths, train_label_paths = download_dataset(path=args.input_path, split="train")
    val_image_paths, val_label_paths = download_dataset(path=args.input_path, split="val")
    test_image_paths, _ = download_dataset(path=args.input_path, split="test")

    # Step 2: Verify the spatial shape of inputs (only for the 'train' split)
    verify_inputs(image_paths=train_image_paths, label_paths=train_label_paths)

    # Step 3: Preprocess input images.
    preprocess_inputs(image_paths=train_image_paths)
    preprocess_inputs(image_paths=val_image_paths)
    preprocess_inputs(image_paths=test_image_paths)

    if args.view:
        # Step 3(a): Visualize the images and corresponding labels (only for the  'train' split)
        visualize_inputs(image_paths=train_image_paths, label_paths=train_label_paths)

    # Step 4: Get the dataloaders.
    train_loader, val_loader = get_dataloaders(
        train_image_paths=train_image_paths,
        train_label_paths=train_label_paths,
        val_image_paths=val_image_paths,
        val_label_paths=val_label_paths,
        view=args.view,
        train_instance_segmentation=train_instance_segmentation,
    )

    # Step 5: Run the finetuning for Segment Anything Model using Ray.
    checkpoint_path = run_finetuning(
        train_loader=train_loader,
        val_loader=val_loader,
        save_root=args.save_root,
        train_instance_segmentation=train_instance_segmentation,
        device=device,
        model_type=model_type,
        overwrite=args.overwrite,
    )

    # Step 6: Run automatic instance segmentation using the finetuned model.
    run_instance_segmentation_with_decoder(
        test_image_paths=test_image_paths, model_type=model_type, checkpoint=checkpoint_path, device='cuda',
    )


if __name__ == "__main__":
    main()