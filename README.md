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
To build the docker image
```
docker build -f Dockerfile -t raysam .
```

To run the docker image
```
docker run -it --shm-size 60G --name raysam --gpus all -v YourLocalPath:/storage/raysam_user/ raysam /bin/bash
```

NOTE: Replace *YourLocalPath* with your own path.

## TODO List
- [ ] Move the dataloaders into the training function
- [ ] Modify the training function ```sam_training.train_sam_worker``` for ray
- [x] Now the training function is working with ray. See the ```debug_sam_finetuning_ray.py```. Refer to the notebook ```sam_finetuning_ray.ipynb``` for data preparation.

NOTE: The ```sam_finetuning_ray``` notebook can now be debugged and run on the local machine within a docker container, with an error triggered by the ```result = trainer.fit()``` as it's not adapted. --Jim 