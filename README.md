# ImageDDS: A Multi-modal Deep Learning Model for Drug Synergy Prediction Using Image and Graph Representation Learning

## Abstract
In synergistic drug combination therapy, computational screening has become a crucial tool for identifying optimal drug combinations. In recent years, graph neural networks (GNNs) have emerged as the primary method for extracting drug features in synergy prediction. However, conventional GNN models still face limitations in capturing complex chemical structures, particularly for cyclic molecules and structures with a large number of atoms. In this work, we propose a multi-modal deep learning framework, ImageDDS, that integrates molecular graphs and images, and leverages self-supervised contrastive learning to align chemical knowledge across both graphs and images for drug synergy prediction. Specifically, we employ a graph Transformer and ResNet18 to extract features from molecular graphs and images, respectively, and fuse these modality-specific features to create a global drug representation. During this process, we further apply contrastive learning to optimize the alignment of graph and image representations, ensuring consistency of features for the same drug across different modalities. Finally, we combine the multi-modal drug features with gene expression embedding from cell lines and feed them into a predictor to identify synergistic drug combinations. In benchmark datasets, we compare ImageDDS with classical machine learning methods and other deep learning-based approaches, with leave-one-out cross-validation demonstrating the superior performance of ImageDDS. Furthermore, we explore the interpretability of graph and image models in discovering important substructures, revealing the crucial role of image features in identifying key drug substructures, especially cyclic structures.
![model](https://github.com/AnQi-87/ImageDDS/blob/main/ImageDDS.png)


## Environment
### create a new conda environment
- conda create -n ImageDDS python=3.8
- conda activate ImageDDS

### install
- pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
- pip install rdkit
- pip install torch==2.3.0+cu121 torchvision==0.14.1+cu121 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
- pip install torch-geometric==1.6.0
- pip install torch-cluster torch-scatter torch-sparse torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.3.0%2Bcu121.html
- pip install seaborn

## Run
- Run the utils_test.py file first

`python utils_test.py`

- Run the image_train.py file then

`python image_train.py --use_image_fusion --use_cl --lambda_cl 0.15 --temperature 0.1 --base_temperature 0.075 --TRAIN_BATCH_SIZE 128 --TEST_BATCH_SIZE 128`
