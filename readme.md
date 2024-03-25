---
runme:
  id: 01HSVVGQ6R93E1HD56WPZ3346G
  version: v3
---

# **Video Colorization Project**

This repository contains code and documentation for our Video Colorization Project.

### Proposed timeline:

April 12 (3 weeks from today): individual pipelines developed; models trained

April 19: training on all data completed; numbers 1-6 in the assignment paper/presentation content written in report

April 26: Rough draft of final report completed with feedback from Caitlyn and/or LaHaye; somewhat decent GUI

May 3: final report completed with subsequent slides based on feedback; GUI finalized

May 3-presentation date: practice presenting, refine as necessary

I was thinking we could set up meetings for each of these checkpoints with further communication as needed. This is just a suggestion so if anyone else has other preferences that’s totally cool as well!


-devyn :,)

## Workspace Setup

```CPSC542_FINAL_PROJECT
├── clips
├── data
│   └── .txt
├── docs
│   └── .txt
├── gray_clips
├── src
│   ├── models
│   │   └── .txt
│   ├── objects
│   │   ├── architecture
│   │   │   └── conv_autoencoder.py
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── result.py
│   │   └── stack.py
│   ├── __init__.py
│   ├── augmentation.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── training.py
│   └── validation.py
├── .gitignore
├── CONTRIBUTING.md
├── main.ipynb
├── readme.md
└── requirements.txt
```
Here's a brief explanation of each file and its role in the project:

- **`readme.md`**
  - Provides an overview of the Video Colorization Project, including a proposed timeline and contributor names.

- **`CONTRIBUTING.md`**
  - Outlines guidelines for contributing to the project, such as branch usage, commit message expectations, and pull request requirements.

- **`main.ipynb`**
  - A Jupyter notebook that serves as the main workspace for the project. It includes sections for preprocessing and augmentation, training, prediction, and validation, with placeholders for code.

- **`src/training.py`**
  - Contains functions for model training and evaluation. It includes a tuner for hyperparameter optimization (`run_tuner`) and a method for training the best model (`get_best_model`). Additionally, it provides a function to evaluate model performance and plot training history.

- **`src/augmentation.py`**
  - Defines an `ImageAugmenter` class responsible for augmenting images. It includes a placeholder method (`augment`) where augmentation techniques can be implemented.

- **`src/objects/stack.py`**
  - Defines a `Stack` class that encapsulates data, model architecture, and results. It includes methods for updating datasets, creating models, and saving the final model and its training history.

- **`src/objects/architecture/conv_autoencoder.py`**
  - Placeholder for the `ConvAutoencoder` class, intended to define the architecture of a convolutional autoencoder model.

- **`src/objects/data.py`**
  - Defines a `Data` class that holds datasets (training, testing, validation) and includes a method for updating these datasets.

- **`src/objects/result.py`**
  - Placeholder for the `Result` class, intended for handling predictions, validation, and visualization tasks such as generating diagrams or applying techniques like GradCAM.

## Contributors

### - Hayden Fargo

### - Tyler Lewis

### - Devyn Miller

### - Ponthea Zahraii
