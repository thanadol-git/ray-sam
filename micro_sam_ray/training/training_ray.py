import os
import time
import warnings
from glob import glob
from tqdm import tqdm
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import imageio.v3 as imageio

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

import ray
import ray.train
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer

import torch_em
from torch_em.data.datasets.util import split_kwargs

from elf.io import open_file

try:
    from qtpy.QtCore import QObject
except Exception:
    QObject = Any

from ..util import get_device
from . import sam_trainer as trainers
from ..instance_segmentation import get_unetr
from . import joint_sam_trainer as joint_trainers
from .util import get_trainable_sam_model, ConvertToSamInputs, require_8bit


FilePath = Union[str, os.PathLike]


# Some functions are copied from the original SAM training script
def _check_input_normalization(x, input_check_done):
        # The expected data range of the SAM model is 8bit (0-255).
        # It can easily happen that data is normalized beforehand in training.
        # For some reasons we don't fully understand this still works, but it
        # should still be avoided and is very detrimental in some settings
        # (e.g. when freezing the image encoder)
        # We check once per epoch if the data seems to be normalized already and
        # raise a warning if this is the case.
        if not input_check_done:
            data_min, data_max = x.min(), x.max()
            if (data_min < 0) or (data_max < 1):
                warnings.warn(
                    "It looks like you are normalizing the training data."
                    "The SAM model takes care of normalization, so it is better to not do this."
                    "We recommend to remove data normalization and input data in the range [0, 255]."
                )
            input_check_done = True

        return input_check_done


def _check_loader(loader, with_segmentation_decoder, name=None, verify_n_labels_in_loader=None):
    x, _ = next(iter(loader))

    # Raw data: check that we have 1 or 3 channels.
    n_channels = x.shape[1]
    if n_channels not in (1, 3):
        raise ValueError(
            "Invalid number of channels for the input data from the data loader. "
            f"Expect 1 or 3 channels, got {n_channels}."
        )

    # Raw data: check that it is between [0, 255]
    minval, maxval = x.min(), x.max()
    if minval < 0 or minval > 255:
        raise ValueError(
            "Invalid data range for the input data from the data loader. "
            f"The input has to be in range [0, 255], but got minimum value {minval}."
        )
    if maxval < 1 or maxval > 255:
        raise ValueError(
            "Invalid data range for the input data from the data loader. "
            f"The input has to be in range [0, 255], but got maximum value {maxval}."
        )

    # Target data: the check depends on whether we train with or without decoder.
    # NOTE: Verification step to check whether all labels from dataloader are valid (i.e. have atleast one instance).

    def _check_instance_channel(instance_channel):
        unique_vals = torch.unique(instance_channel)
        if (unique_vals < 0).any():
            raise ValueError(
                "The target channel with the instance segmentation must not have negative values."
            )
        if len(unique_vals) == 1:
            raise ValueError(
                "The target channel with the instance segmentation must have at least one instance."
            )
        if not torch.allclose(unique_vals, unique_vals.round(), atol=1e-7):
            raise ValueError(
                "All values in the target channel with the instance segmentation must be integer."
            )

    counter = 0
    name = "" if name is None else f"'{name}'"
    for x, y in tqdm(
        loader,
        desc=f"Verifying labels in {name} dataloader",
        total=verify_n_labels_in_loader if verify_n_labels_in_loader is not None else None,
    ):
        n_channels_y = y.shape[1]
        if with_segmentation_decoder:
            if n_channels_y != 4:
                raise ValueError(
                    "Invalid number of channels in the target data from the data loader. "
                    "Expect 4 channel for training with an instance segmentation decoder, "
                    f"but got {n_channels_y} channels."
                )
            # Check instance channel per sample in a batch
            for per_y_sample in y:
                _check_instance_channel(per_y_sample[0])

            targets_min, targets_max = y[:, 1:].min(), y[:, 1:].max()
            if targets_min < 0 or targets_min > 1:
                raise ValueError(
                    "Invalid value range in the target data from the value loader. "
                    "Expect the 3 last target channels (for normalized distances and foreground probabilities)"
                    f"to be in range [0.0, 1.0], but got min {targets_min}"
                )
            if targets_max < 0 or targets_max > 1:
                raise ValueError(
                    "Invalid value range in the target data from the value loader. "
                    "Expect the 3 last target channels (for normalized distances and foreground probabilities)"
                    f"to be in range [0.0, 1.0], but got max {targets_max}"
                )

        else:
            if n_channels_y != 1:
                raise ValueError(
                    "Invalid number of channels in the target data from the data loader. "
                    "Expect 1 channel for training without an instance segmentation decoder,"
                    f"but got {n_channels_y} channels."
                )
            # Check instance channel per sample in a batch
            for per_y_sample in y:
                _check_instance_channel(per_y_sample)

        counter += 1
        if verify_n_labels_in_loader is not None and counter > verify_n_labels_in_loader:
            break


# Make the progress bar callbacks compatible with a tqdm progress bar interface.
class _ProgressBarWrapper:
    def __init__(self, signals):
        self._signals = signals
        self._total = None

    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, value):
        self._signals.pbar_total.emit(value)
        self._total = value

    def update(self, steps):
        self._signals.pbar_update.emit(steps)

    def set_description(self, desc, **kwargs):
        self._signals.pbar_description.emit(desc)


def _count_parameters(model_parameters):
    params = sum(p.numel() for p in model_parameters if p.requires_grad)
    params = params / 1e6
    print(f"The number of trainable parameters for the provided model is {round(params, 2)}M")


@contextmanager
def _filter_warnings(ignore_warnings):
    if ignore_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    else:
        with nullcontext():
            yield

def train_sam_worker(train_config: Dict):
    """Worker function for distributed SAM training using Ray."""

    # Unpack config
    name = train_config["name"]
    model_type = train_config["model_type"]
    train_loader = train_config["train_loader"]     # DataLoader should be initialized in the function
    val_loader = train_config["val_loader"]
    n_epochs = train_config.get("n_epochs", 100)
    early_stopping = train_config.get("early_stopping", 10)
    n_objects_per_batch = train_config.get("n_objects_per_batch", 25)
    checkpoint_path = train_config.get("checkpoint_path", None)
    with_segmentation_decoder = train_config.get("with_segmentation_decoder", True)
    freeze = train_config.get("freeze", None)
    # device = train_config.get("device", None)
    lr = train_config.get("lr", 1e-4)
    wd = train_config.get("wd", 1e-2)
    n_sub_iteration = train_config.get("n_sub_iteration", 8)
    save_root = train_config.get("save_root", None)
    mask_prob = train_config.get("mask_prob", 0.5)
    n_iterations = train_config.get("n_iterations", None)
    scheduler_class = train_config.get("scheduler_class", torch.optim.lr_scheduler.ReduceLROnPlateau)
    scheduler_kwargs = train_config.get("scheduler_kwargs", None)
    optimizer_class = train_config.get("optimizer_class", torch.optim.AdamW)
    peft_kwargs = train_config.get("peft_kwargs", None)
    model_kwargs = train_config.get("model_kwargs", {})

    t_start = time.time()

    # Check loaders
    # _check_loader(train_loader, with_segmentation_decoder, "train")
    # _check_loader(val_loader, with_segmentation_decoder, "val")

    # Prepare data loaders for distributed training
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    val_loader = ray.train.torch.prepare_data_loader(val_loader)

    device = ray.train.torch.get_device()

    # Get the trainable segment anything model
    model, state = get_trainable_sam_model(
        model_type=model_type,
        device=device,
        freeze=freeze,
        checkpoint_path=checkpoint_path,
        return_state=True,
        peft_kwargs=peft_kwargs,
        **model_kwargs
    )

    # Create inputs converter
    convert_inputs = ConvertToSamInputs(transform=model.transform, box_distortion_factor=0.025)
    
    # Set up optimizer and scheduler kwargs
    if scheduler_kwargs is None:
        scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 2, "verbose": True}

    # Handle segmentation decoder case
    if with_segmentation_decoder:
        # Get the UNETR
        unetr = get_unetr(
            image_encoder=model.sam.image_encoder,
            decoder_state=state.get("decoder_state", None),
            device=device,
        )
        
        # Get the parameters for SAM and the decoder from UNETR
        joint_model_params = [params for params in model.parameters()]  # sam parameters
        for param_name, params in unetr.named_parameters():  # unetr's decoder parameters
            if not param_name.startswith("encoder"):
                joint_model_params.append(params)

        optimizer = optimizer_class(joint_model_params, lr=lr, weight_decay=wd, amsgrad=True)
        scheduler = scheduler_class(optimizer=optimizer, **scheduler_kwargs)

        # Set up trainer with instance segmentation
        instance_seg_loss = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=True)
        trainer = joint_trainers.JointSamTrainer(
            name=name,
            save_root=save_root,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            lr_scheduler=scheduler,
            logger=joint_trainers.JointSamLogger,
            log_image_interval=100,
            mixed_precision=True,
            convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch,
            n_sub_iteration=n_sub_iteration,
            compile_model=False,
            unetr=unetr,
            instance_loss=instance_seg_loss,
            instance_metric=instance_seg_loss,
            early_stopping=early_stopping,
            mask_prob=mask_prob,
            use_ray=True,
        )
    else:
        # Set up standard SAM trainer
        optimizer = optimizer_class(model.parameters(), lr=lr)
        scheduler = scheduler_class(optimizer=optimizer, **scheduler_kwargs)
        
        trainer = trainers.SamTrainer(
            name=name,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            lr_scheduler=scheduler,
            logger=trainers.SamLogger,
            log_image_interval=100,
            mixed_precision=True,
            convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch,
            n_sub_iteration=n_sub_iteration,
            compile_model=False,
            early_stopping=early_stopping,
            mask_prob=mask_prob,
            save_root=save_root,
            use_ray=True,
        )

    # Set up training parameters
    if n_iterations is None:
        trainer_fit_params = {"epochs": n_epochs}
    else:
        trainer_fit_params = {"iterations": n_iterations}

    # Prepare model for distributed training
    model = ray.train.torch.prepare_model(model)
    unetr = ray.train.torch.prepare_model(unetr)
    
    trainer.model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.unetr = unetr
    # Training loop
    best_metric = float("inf")
    for epoch in range(n_epochs):
        if ray.train.get_context().get_world_size() > 1:
            # Required for distributed sampling
            train_loader.sampler.set_epoch(epoch)
        
        # Run training iteration
        # TODO: training is done by DefaultTrainer.fit() method. We need to run it here for each epoch.
        # train_metrics = trainer.train_iteration()
        # val_metrics = trainer.validate()
        
        """Still debugging, start"""
        # Run training iteration
        model.train()
        
        input_check_done = False
        n_iter = 0
        loss_train, loss_unetr_train, model_iou_train = 0.0, 0.0, 0.0
        t_per_iter = time.time()
        for x, y in train_loader:
            labels_instances = y[:, 0, ...].unsqueeze(1)
            labels_for_unetr = y[:, 1:, ...]

            input_check_done = _check_input_normalization(x, input_check_done)

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda"):
                # 1. train for the interactive segmentation
                (loss, mask_loss, iou_regression_loss, model_iou,
                 sampled_binary_y) = trainer._interactive_train_iteration(x, labels_instances)

            # backprop(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda"):
                # 2. train for the automatic instance segmentation
                unetr_loss = trainer._instance_iteration(x, labels_for_unetr)

            # backprop(unetr_loss)
            unetr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_train += loss.item()
            loss_unetr_train += unetr_loss.item()
            model_iou_train += model_iou.item()

            # if self.logger is not None:
            #     lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
            #     samples = sampled_binary_y if self._iteration % self.log_image_interval == 0 else None
            #     self.logger.log_train(
            #         self._iteration, loss, lr, x, labels_instances, samples,
            #         mask_loss, iou_regression_loss, model_iou, unetr_loss
            #     )

            n_iter += 1
            # if self._iteration >= self.max_iteration:
            #     break
            # progress.update(1)

        loss_train /= len(val_loader)
        loss_unetr_train /= len(val_loader)
        model_iou_train /= len(val_loader)
        t_per_iter = (time.time() - t_per_iter) / n_iter
        metrics_train = {
            "epoch": epoch,
            "loss": loss_train,
            "instance_loss": loss_unetr_train,
            "model_iou": model_iou_train,
            "time_per_iter": t_per_iter
        }
        
        # Run validation
        model.eval()

        input_check_done = False

        val_iteration = 0
        metric_val, loss_val, unetr_loss_val, model_iou_val = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                labels_instances = y[:, 0, ...].unsqueeze(1)
                labels_for_unetr = y[:, 1:, ...]

                input_check_done = _check_input_normalization(x, input_check_done)

                with torch.autocast(device_type="cuda"):
                    # 1. validate for the interactive segmentation
                    (loss, mask_loss, iou_regression_loss, model_iou,
                     sampled_binary_y, metric) = trainer._interactive_val_iteration(x, labels_instances, val_iteration)

                with torch.autocast(device_type="cuda"):
                    # 2. validate for the automatic instance segmentation
                    unetr_loss, unetr_metric = trainer._instance_iteration(x, labels_for_unetr, metric_for_val=True)

                loss_val += loss.item()
                unetr_loss_val += unetr_loss.item()
                metric_val += metric.item() + (unetr_metric.item() / 3)
                model_iou_val += model_iou.item()
                val_iteration += 1

        loss_val /= len(val_loader)
        metric_val /= len(val_loader)
        model_iou_val /= len(val_loader)
        metrics_val = {
            "epoch": epoch,
            "loss": loss_val,
            "instance_loss": unetr_loss_val,
            "model_iou": model_iou_val,
            "metric_val": metric_val
        }

        # if self.logger is not None:
        #     self.logger.log_validation(
        #         self._iteration, metric_val, loss_val, x, labels_instances, sampled_binary_y,
        #         mask_loss, iou_regression_loss, model_iou_val, unetr_loss
        #     )
                
        # Report metrics to Ray
        metrics = {
            "epoch": epoch,
            "train": metrics_train,
            "val": metrics_val
        }

        # save_checkpoint
        if metric_val < best_metric:
            best_metric = metric_val
            best_epoch = epoch
            if save_root is not None:
                checkpoint_dir = os.path.join(save_root, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_best.pt")
                current_unetr_state = unetr.state_dict()
                decoder_state = []
                for k, v in current_unetr_state.items():
                    if not k.startswith("encoder"):
                        decoder_state.append((k, v))
                decoder_state = OrderedDict(decoder_state)

                torch.save({
                    'best_epoch': best_epoch,
                    'model_state': model.state_dict(),
                    'decoder_state': decoder_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics_train': metrics_train,
                    'metrics_val': metrics_val,
                    }, checkpoint_path)
                checkpoint_ray = Checkpoint.from_directory(checkpoint_dir)
        
        # Report metrics to Ray
        ray.train.report(metrics, checkpoint=checkpoint_ray)
        
        # Handle early stopping 
        # TODO: not implemented yet
        # if trainer.early_stopping is not None:
        #     if trainer.early_stopping.should_stop():
        #         print(f"Early stopping triggered at epoch {epoch}")
        #         break

    # Print training time
    t_run = time.time() - t_start
    hours = int(t_run // 3600)
    minutes = int((t_run % 3600) // 60)
    seconds = int(round(t_run % 60, 0))
    print(f"Training took {t_run} seconds (= {hours:02}:{minutes:02}:{seconds:02} hours)")


def train_sam_distributed(
    name: str,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_workers: int = 2,
    use_gpu: bool = True,
    **train_kwargs
    ) -> None:
    """Distributed training wrapper for SAM using Ray."""

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    # Prepare training configuration
    train_config = {
        "name": name,
        "model_type": model_type,
        "train_loader": train_loader,
        "val_loader": val_loader,
        **train_kwargs
    }

    # Configure computation resources
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu
    )

    # Initialize the Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_sam_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
    )

    # Start distributed training
    result = trainer.fit()
    print(f"Training completed with result: {result}")
  
  
def _update_patch_shape(patch_shape, raw_paths, raw_key, with_channels):
    if isinstance(raw_paths, (str, os.PathLike)):
        path = raw_paths
    else:
        path = raw_paths[0]
    assert isinstance(path, (str, os.PathLike))

    # Check the underlying data dimensionality.
    if raw_key is None:  # If no key is given then we assume it's an image file.
        ndim = imageio.imread(path).ndim
    else:  # Otherwise we try to open the file from key.
        try:  # First try to open it with elf.
            with open_file(path, "r") as f:
                ndim = f[raw_key].ndim
        except ValueError:  # This may fail for images in a folder with different sizes.
            # In that case we read one of the images.
            image_path = glob(os.path.join(path, raw_key))[0]
            ndim = imageio.imread(image_path).ndim

    if ndim == 2:
        assert len(patch_shape) == 2
        return patch_shape
    elif ndim == 3 and len(patch_shape) == 2 and not with_channels:
        return (1,) + patch_shape
    elif ndim == 4 and len(patch_shape) == 2 and with_channels:
        return (1,) + patch_shape
    else:
        return patch_shape


def default_sam_dataset(
    raw_paths: Union[List[FilePath], FilePath],
    raw_key: Optional[str],
    label_paths: Union[List[FilePath], FilePath],
    label_key: Optional[str],
    patch_shape: Tuple[int],
    with_segmentation_decoder: bool,
    with_channels: bool = False,
    sampler: Optional[Callable] = None,
    raw_transform: Optional[Callable] = None,
    n_samples: Optional[int] = None,
    is_train: bool = True,
    min_size: int = 25,
    max_sampling_attempts: Optional[int] = None,
    **kwargs,
) -> Dataset:
    """Create a PyTorch Dataset for training a SAM model.

    Args:
        raw_paths: The path(s) to the image data used for training.
            Can either be multiple 2D images or volumetric data.
        raw_key: The key for accessing the image data. Internal filepath for hdf5-like input
            or a glob pattern for selecting multiple files.
        label_paths: The path(s) to the label data used for training.
            Can either be multiple 2D images or volumetric data.
        label_key: The key for accessing the label data. Internal filepath for hdf5-like input
            or a glob pattern for selecting multiple files.
        patch_shape: The shape for training patches.
        with_segmentation_decoder: Whether to train with additional segmentation decoder.
        with_channels: Whether the image data has RGB channels.
        sampler: A sampler to reject batches according to a given criterion.
        raw_transform: Transformation applied to the image data.
            If not given the data will be cast to 8bit.
        n_samples: The number of samples for this dataset.
        is_train: Whether this dataset is used for training or validation.
        min_size: Minimal object size. Smaller objects will be filtered.
        max_sampling_attempts: Number of sampling attempts to make from a dataset.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """

    # Set the data transformations.
    if raw_transform is None:
        raw_transform = require_8bit

    if with_segmentation_decoder:
        label_transform = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=min_size,
        )
    else:
        label_transform = torch_em.transform.label.MinSizeLabelTransform(
            min_size=min_size
        )

    # Set a default sampler if none was passed.
    if sampler is None:
        sampler = torch_em.data.sampler.MinInstanceSampler(3, min_size=min_size)

    # Check the patch shape to add a singleton if required.
    patch_shape = _update_patch_shape(
        patch_shape, raw_paths, raw_key, with_channels
    )

    # Set a minimum number of samples per epoch.
    if n_samples is None:
        loader = torch_em.default_segmentation_loader(
            raw_paths=raw_paths,
            raw_key=raw_key,
            label_paths=label_paths,
            label_key=label_key,
            batch_size=1,
            patch_shape=patch_shape,
            ndim=2,
            **kwargs
        )
        n_samples = max(len(loader), 100 if is_train else 5)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=raw_key,
        label_paths=label_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        raw_transform=raw_transform,
        label_transform=label_transform,
        with_channels=with_channels,
        ndim=2,
        sampler=sampler,
        n_samples=n_samples,
        **kwargs,
    )

    if max_sampling_attempts is not None:
        if isinstance(dataset, torch_em.data.concat_dataset.ConcatDataset):
            for ds in dataset.datasets:
                ds.max_sampling_attempts = max_sampling_attempts
        else:
            dataset.max_sampling_attempts = max_sampling_attempts

    return dataset


# def default_sam_loader_distributed(**kwargs) -> DataLoader:
#     """Create a distributed DataLoader for training SAM with Ray."""
#     sam_ds_kwargs, extra_kwargs = split_kwargs(default_sam_dataset, **kwargs)

#     # Get dataset
#     ds = default_sam_dataset(**sam_ds_kwargs)

#     # Add DistributedSampler for distributed training
#     # sampler = torch.utils.data.DistributedSampler(ds)

#     # Update loader kwargs to use the distributed sampler
#     loader_kwargs = {
#         # "sampler": sampler,
#         "shuffle": True,  # Shuffle is handled by DistributedSampler
#         **extra_kwargs
#     }

#     return torch_em.segmentation.get_data_loader(ds, **loader_kwargs)

def default_sam_loader_distributed(**kwargs) -> DataLoader:
    """Create a PyTorch DataLoader for training a SAM model in a distributed setup.

    Args:
        kwargs: Keyword arguments for `micro_sam.training.default_sam_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    # Split the arguments for the dataset and the DataLoader
    sam_ds_kwargs, extra_kwargs = split_kwargs(default_sam_dataset, **kwargs)

    # Further split the arguments for the segmentation dataset and the DataLoader
    extra_ds_kwargs, loader_kwargs = split_kwargs(torch_em.default_segmentation_dataset, **extra_kwargs)

    # Combine the dataset-specific arguments
    ds_kwargs = {**sam_ds_kwargs, **extra_ds_kwargs}

    # Ensure shuffle is set for the DataLoader
    loader_kwargs.setdefault("shuffle", True)

    # Create the dataset
    ds = default_sam_dataset(**ds_kwargs)

    # Create the DataLoader
    return torch_em.segmentation.get_data_loader(ds, **loader_kwargs)

CONFIGURATIONS = {
    "Minimal": {"model_type": "vit_t", "n_objects_per_batch": 4, "n_sub_iteration": 4},
    "CPU": {"model_type": "vit_b", "n_objects_per_batch": 10},
    "gtx1080": {"model_type": "vit_t", "n_objects_per_batch": 5},
    "rtx5000": {"model_type": "vit_b", "n_objects_per_batch": 10},
    "V100": {"model_type": "vit_b"},
    "A100": {"model_type": "vit_h"},
}
"""Best training configurations for given hardware resources.
"""


def train_sam_for_configuration(
    name: str,
    configuration: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    with_segmentation_decoder: bool = True,
    model_type: Optional[str] = None,
    **kwargs,
) -> None:
    """Run training for a SAM model with the configuration for a given hardware resource.

    Selects the best training settings for the given configuration.
    The available configurations are listed in `CONFIGURATIONS`.

    Args:
        name: The name of the model to be trained.
            The checkpoint and logs wil have this name.
        configuration: The configuration (= name of hardware resource).
        train_loader: The dataloader for training.
        val_loader: The dataloader for validation.
        checkpoint_path: Path to checkpoint for initializing the SAM model.
        with_segmentation_decoder: Whether to train additional UNETR decoder
            for automatic instance segmentation.
        model_type: Over-ride the default model type.
            This can be used to use one of the micro_sam models as starting point
            instead of a default sam model.
        kwargs: Additional keyword parameters that will be passed to `train_sam`.
    """
    if configuration in CONFIGURATIONS:
        train_kwargs = CONFIGURATIONS[configuration]
    else:
        raise ValueError(f"Invalid configuration {configuration} expect one of {list(CONFIGURATIONS.keys())}")

    if model_type is None:
        model_type = train_kwargs.pop("model_type")
    else:
        expected_model_type = train_kwargs.pop("model_type")
        if model_type[:5] != expected_model_type:
            warnings.warn("You have specified a different model type.")

    train_kwargs.update(**kwargs)
    train_sam_distributed(
        name=name, train_loader=train_loader, val_loader=val_loader,
        checkpoint_path=checkpoint_path, with_segmentation_decoder=with_segmentation_decoder,
        model_type=model_type, **train_kwargs
    )
