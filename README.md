
![kaggle header](https://github.com/user-attachments/assets/79e41d49-3d9a-4960-8ee0-22d3e14b4dd7)
This is the project repository for **WASP Scalable Data Science and Distributed Machine Learning 2024 Group 3**.  

In this project, we fine-tuned the **Segment Anything Model (SAM)**, an advanced vision foundation model developed by Meta AI. SAM is designed for promptable segmentation tasks and is a highly versatile tool for image segmentation across diverse domains. For more information, you can explore the [SAM GitHub repository](https://github.com/facebookresearch/segment-anything) and the accompanying paper, ["Segment Anything"](https://arxiv.org/abs/2304.02643).  

We built on the fine-tuning approach demonstrated by the [micro SAM repository](https://github.com/computational-cell-analytics/micro-sam), which specializes in adapting SAM to fluorescence microscopy datasets. Our project introduces a novel contribution by leveraging the **Ray framework** to enable scalable, distributed training of SAM, making it suitable for handling large-scale microscopy datasets.  

To achieve this, we utilized **Ray Train** and its [`TorchTrainer`](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html) module. The `TorchTrainer` is a tool designed for data-parallel PyTorch training, automating the setup of distributed environments for scalable execution. It launches multiple workers as specified in the scaling configuration, establishes a distributed PyTorch environment for those workers, and seamlessly ingests input datasets. Each worker executes the user-defined `train_loop_per_worker` function, which contains the core training logic. This framework allowed us to scale SAM fine-tuning efficiently across multiple nodes, making it highly adaptable to large microscopy datasets.  

This repository includes the code, configuration files, and documentation required to reproduce our results and experiment further with distributed fine-tuning of SAM.

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
docker run -it --shm-size 20G --gpus all -v YOUR_LOCAL_PATH:/storage/raysam_user/ -e RAY_TMPDIR=/storage/raysam_user/tmp raysam:2.24.0
```
To run the code:
```
# Go to the project directory
cd /storage/raysam_user/ray-sam

# Run the demo
python demo_sam_finetuning_ray.py

# Run with the Human Protein Atlas(HPA) dataset:

python run_sam_finetuning_hpa.py
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
- Jingyu Guo
  - Training and Evaluation Scripts with Ray: Developed training and evaluation scripts using the micro-sam and torch-em frameworks, which were further optimized for seamless integration with Ray to ensure scalability and efficiency. Conducted thorough debugging and testing on both local machines and servers to guarantee reliability and robustness.
  - Docker Image: Developed a Docker image specifically tailored to meet the project’s requirements, ensuring that all dependencies are effectively managed.
  - Jupyter Notebooks: Designed and implemented the Jupyter notebooks to enable scalable fine-tuning of models. 
  - Group Discussions and Presentation: Actively engaged in group discussions to share insights, gave and received feedback, and contributed to the final presentation.

- Nils Mechtel: Ray system, and Jupyter notebooks
- Songtao Cheng: 
    -   **Dataset**: Finding the Human Protein Atlas (HPA) dataset with annotated masks for use in the project.
    -   **Training and Evaluation Scripts with Ray**: Modified the `micro_sam` and `torch_em` frameworks to ensure compatibility with Ray. Developed code for fine-tuning the SAM model using the HPA dataset.
    -   **Docker Image**: Contributed to improving the Docker image by debugging, optimizing, and reducing its file size. To make sure the image could be built and published on GitHub.
    -   **Group Meetings and Discussions**: Attended meetings and followed up on project progress.
- Thanadol Sutantiwanichkul: 
  - Contacting the Human Protein Atlas (HPA) IT: Established communication with the HPA IT team to obtain the necessary resources for the project, including pre-trained images and masks.
  - Weekly Meetings: Participated in weekly meetings to discuss project progress, share insights, and provide feedback to the team.
  - Code Implementation: Conducted code implementation and training to contribute to the project’s development.
  - Discussion and review of the project: attending the meeting and follow up on the project progress.