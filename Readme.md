# AI Programming with Python - Image Classifier Project

## Project Overview
This project is an image classification application that can train a deep learning model on a dataset of images and then predict the classes for new images. It includes both a Jupyter notebook for development and a command-line application.

## Project Structure
```
.
├── Image Classifier Project.ipynb
├── Image Classifier Project.html
├── train.py
├── predict.py
├── model.py
├── utils.py
├── cat_to_name.json
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Training
Train a new network on a data set:
```bash
python train.py data_directory
```

#### Options:
* Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
* Choose architecture: `python train.py data_dir --arch "vgg13"`
* Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
* Use GPU for training: `python train.py data_dir --gpu`

### Prediction
Predict flower name from an image:
```bash
python predict.py /path/to/image checkpoint
```

#### Options:
* Return top K most likely classes: `python predict.py input checkpoint --top_k 3`
* Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
* Use GPU for inference: `python predict.py input checkpoint --gpu`

## Data
The dataset used for training is not included in the repository. The model was trained on a flower dataset with 102 different categories.
