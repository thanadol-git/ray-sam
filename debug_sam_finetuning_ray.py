# Check if we are running this notebook on kaggle, google colab or local compute resources.
import os
import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import warnings

warnings.filterwarnings("once")

from glob import glob
from IPython.display import FileLink

import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components

import torch

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.util.util import get_random_colors

import sys

# sys.path.append('/home/scheng/workspace/RaySam/micro-sam-ray')
import micro_sam_ray.training as sam_training
from micro_sam_ray.sample_data import (
    fetch_tracking_example_data,
    fetch_tracking_segmentation_data,
)
from micro_sam_ray.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)


def run_automatic_instance_segmentation(
    image, checkpoint_path, model_type="vit_b_lm", device=None
):
    """Automatic Instance Segmentation (AIS) by training an additional instance decoder in SAM.

    NOTE: AIS is supported only for `µsam` models.

    Args:
        image: The input image.
        checkpoint_path: The path to stored checkpoints.
        model_type: The choice of the `µsam` model.
        device: The device to run the model inference.

    Returns:
        The instance segmentation.
    """
    # Step 1: Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,  # choice of the Segment Anything model
        checkpoint=checkpoint_path,  # overwrite to pass your own finetuned model.
        device=device,  # the device to run the model inference.
    )

    # Step 2: Get the instance segmentation for the given image.
    prediction = automatic_instance_segmentation(
        predictor=predictor,  # the predictor for the Segment Anything model.
        segmenter=segmenter,  # the segmenter class responsible for generating predictions.
        input_path=image,
        ndim=2,
    )

    return prediction


def debug_notebook(training=True, inference=False):
    root_dir = "/storage/raysam_user/GitHub/RaySam"  # overwrite to set the root directory, where the data, checkpoints, and all relevant stuff will be stored

    DATA_FOLDER = os.path.join(root_dir, "data")
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # This will download the image and segmentation data for training.
    image_dir = fetch_tracking_example_data(DATA_FOLDER)
    segmentation_dir = fetch_tracking_segmentation_data(DATA_FOLDER)

    image_paths = sorted(glob(os.path.join(image_dir, "*")))
    segmentation_paths = sorted(glob(os.path.join(segmentation_dir, "*")))

    # Load images from multiple files in folder via pattern (here: all tif files)
    raw_key, label_key = "*.tif", "*.tif"

    # Alternative: if you have tif stacks you can just set 'raw_key' and 'label_key' to None
    # raw_key, label_key= None, None

    # The 'roi' argument can be used to subselect parts of the data.
    # Here, we use it to select the first 70 images (frames) for the train split and the other frames for the val split.
    train_roi = np.s_[:70, :, :]
    val_roi = np.s_[70:, :, :]

    batch_size = 1  # the training batch size
    patch_shape = (1, 512, 512)  # the size of patches for training

    train_instance_segmentation = True

    # There are cases where our inputs are large and the labeled objects are not evenly distributed across the image.
    # For this we use samplers, which ensure that valid inputs are chosen subjected to the paired labels.
    # The sampler chosen below makes sure that the chosen inputs have atleast one foreground instance, and filters out small objects.
    sampler = MinInstanceSampler(
        min_size=25
    )  # NOTE: The choice of 'min_size' value is paired with the same value in 'min_size' filter in 'label_transform'.

    # Update the train_loader and val_loader creation to use DistributedSampler
    train_loader = sam_training.training_ray.default_sam_loader_distributed(
        raw_paths=image_dir,
        raw_key=raw_key,
        label_paths=segmentation_dir,
        label_key=label_key,
        with_segmentation_decoder=train_instance_segmentation,
        patch_shape=patch_shape,
        batch_size=batch_size,
        sampler=sampler,
    )

    val_loader = sam_training.training_ray.default_sam_loader_distributed(
        raw_paths=image_dir,
        raw_key=raw_key,
        label_paths=segmentation_dir,
        label_key=label_key,
        with_segmentation_decoder=train_instance_segmentation,
        patch_shape=patch_shape,
        batch_size=batch_size,
        sampler=sampler,
    )

    # NOTE (Jim): These loaders should be initialized in the train function, not here.

    # All hyperparameters for training
    n_objects_per_batch = 5  # the number of objects per batch that will be sampled
    n_epochs = 1  # how long we train (in epochs)
    model_type = "vit_b"  # using vit_b for faster training
    checkpoint_name = "sam_hela"

    if training:
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Configure the scaling for distributed training
        scaling_config = ScalingConfig(
            num_workers=2,  # number of worker processes
            use_gpu=True,  # use GPU
            resources_per_worker={
                "CPU": 8,  # limit CPU cores per worker
                "GPU": 1,  # each worker gets 2 GPUs
            },
        )

        # NOTE: We should avoid passing the dataloaders to the training config, as suggested by ray.
        train_config = {
            "name": checkpoint_name,
            "save_root": os.path.join(root_dir, "saved_runs"),
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

        # Initialize the distributed trainer
        trainer = TorchTrainer(
            train_loop_per_worker=sam_training.train_sam_worker,
            train_loop_config=train_config,
            scaling_config=scaling_config,
            run_config=ray.train.RunConfig(
                storage_path="/storage/raysam_user/tmp/debug_sam_finetuning_ray",
                name="debug_sam_finetuning_ray",
            ),
        )

        # Run distributed training
        result = trainer.fit()
        print(f"Training completed with result: {result}")

    best_checkpoint = os.path.join(
        root_dir, "saved_runs", "checkpoints", "checkpoint_best.pt"
    )

    if inference:
        assert os.path.exists(
            best_checkpoint
        ), "Please train the model first to run inference on the finetuned model."
        assert (
            train_instance_segmentation is True
        ), "Oops. You didn't opt for finetuning using the decoder-based automatic instance segmentation."

        # Let's check the first 5 images. Feel free to comment out the line below to run inference on all images.
        image_paths = image_paths[:5]

        for idx, image_path in enumerate(image_paths):
            image = imageio.imread(image_path)

            # Predicted instances
            prediction = run_automatic_instance_segmentation(
                image=image,
                checkpoint_path=best_checkpoint,
                model_type=model_type,
                device="cuda",
            )

            # Visualize the predictions
            fig, ax = plt.subplots(1, 2, figsize=(10, 10))

            ax[0].imshow(image, cmap="gray")
            ax[0].axis("off")
            ax[0].set_title("Input Image")

            ax[1].imshow(
                prediction, cmap=get_random_colors(prediction), interpolation="nearest"
            )
            ax[1].axis("off")
            ax[1].set_title("Predictions (AIS)")

            plt.show()
            # Save the figure
            plt.savefig(
                os.path.join(root_dir, "saved_runs", f"checkpoint_best_img{idx}.png")
            )
            plt.close()


if __name__ == "__main__":
    debug_notebook(training=True, inference=True)
