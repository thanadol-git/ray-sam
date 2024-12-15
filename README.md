# RaySam
This is the project repository for WASP Scalable Data Science and Distributed Machine Learning 2024. Our group want to apply the distributed machine learning to the ray systems that will be available in the future at the Science for Life Laboratory (SciLifeLab) in Stockholm. The dataset will be mostly florescent microscopy images and the goal is to train a model to predict the cell type of the images. We plan to scale in different ways, such as scaling the number of images, scaling the number of nodes, and scaling the number of GPUs. 

## Project Structure
- `src/` contains the source code for the project
- `data/` contains the dataset
- `notebooks/` contains the notebooks for the project
- `results/` contains the results of the project
- `scripts/` contains the scripts for the project
- `docs/` contains the documentation for the project
- `requirements.yml` contains the required packages for the project


## Teams 
- Thanadol Sutantiwanichkul
- Songtao Cheng
- Nils Mechtel 
- Jingyu Guo

## Project Plan

## Dataset

## Installation 

### Docker 
To download the docker image
```
docker pull ghcr.io/thanadol-git/ray-sam:latest
```

To run the docker image
```
docker run -it --shm-size 20G --gpus all \
  -v YOUR_LOCAL_PATH:/storage/raysam_user/ \
  -e RAY_TMPDIR=/storage/raysam_user/tmp \
  raysam:2.24.0
```
To run the code:
```
# In the docker image:
cd /storage/raysam_user/
```

NOTE: Replace *YourLocalPath* with your own path.

## TODO List
- [x] Move the dataloaders into the training function
- [x] Modify the training function ```sam_training.train_sam_worker``` for ray
- [x] Now the training function is working with ray. See the ```debug_sam_finetuning_ray.py```. Refer to the notebook ```sam_finetuning_ray.ipynb``` for data preparation.
- [x] Automatically download, preprocess, finetuning model, and run automatic instance segmentation using the finetuned model and test data. See the ```run_sam_finetuning_hpa.py```

NOTE: The ```sam_finetuning_ray``` notebook can now be debugged and run on the local machine within a docker container, with an error triggered by the ```result = trainer.fit()``` as it's not adapted. --Jim 
