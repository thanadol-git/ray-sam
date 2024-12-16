![raysam_header](https://github.com/user-attachments/assets/f22662d4-b2ac-446a-9142-7e0d15e8be97)

This is the project repository for WASP Scalable Data Science and Distributed Machine Learning 2024. Our group want to apply the distributed machine learning to the ray systems that will be available in the future at the Science for Life Laboratory (SciLifeLab) in Stockholm. Here, we will uuse the state-of-the-art deep learning model for image segmentation, called Spatially Adaptive Denormalization (SAM), to predict the cell type of the images. It has been eailier implemented in several bioimages with the tool called micro-SAM. We will use the ray system to scale the training of the model to a large dataset. This can be easily implemented with standard jupyter notebooks and python scripts. One can also use docker image to run the code. We plan to scale in different ways, such as scaling the number of images, scaling the number of nodes, and scaling the number of GPUs. 

## Team Members
- Jingyu Guo (jingyug@kth.se) $^{1}$
- Nils Mechtel (mechtel@kth.se) $^{2}$
- Songtao Cheng (songtaoc@kth.se) $^{2}$
- Thanadol Sutantiwanichkul (thanadol@kth.se) $^{3}$

$^{1}$ Division of Computational Science and Technology, School of Electrical Engineering and Computer Science, KTH Royal Institute of Technology

$^{2}$ Division of Applied Physics, School of Engineering Sciences, KTH Royal Institute of Technology

$^{3}$ Division of Systems Biology, School of Engineering Sciences in Chemistry, Biotechnology and Health, KTH Royal Institute of Technology

## Link to documents 
Link to the [presentation](https://docs.google.com/presentation/d/1KyzPKBo25B9-GNr_semnD0oxbj-Y88YiK_fBCCQF9fQ/edit?usp=sharing)


## Turotials 

### Ray cluster
A general introduction to Raz clusters can be found in the [Ray documentation](https://docs.ray.io/en/latest/cluster/getting-started.html).

To deploy a Ray cluster on Kubernetes using the KubeRay project, follow the instruction in [KubeRay Readme](kuberay-cluster/).

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

NOTE: Replace *YOUR_LOCAL_PATH* with your own path.

### Jupyter notebooks

Create Conda Environment
```
bash create_conda_env.sh
```

Activate Conda Environment
```
conda activate raysam
```

ETC. 

## Contribution 
- Jingyu Guo: Docker image, Ray system, and Jupyter notebooks
- Nils Mechtel: Ray system, and Jupyter notebooks
- Songtao Cheng: Ray system, and Jupyter notebooks
- Thanadol Sutantiwanichkul: 
`````````
In the first/second weeks, I took a lead on data selection and preprocessing. I went to contact the HPA IT in order to use the most possible resources for our project. 
```
