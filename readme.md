## CPSC 542 Final Project

# Video Colorization Project

This repository contains code and documentation for our Video Colorization Project.

## Workspace Setup

```sh
video_colorization_project/
│
├── docs/                   # Documentation files
│   ├── conf.py
│   └── index.rst
│
├── models/                 # Autoencoder and GAN model definitions
│   ├── autoencoder.py
│   └── gan.py
│
├── notebooks/              # Jupyter notebooks for exploration and presentation
│
├── src/                    # Source code for use in this project
│   ├── __init__.py         # Makes src a Python module
│   ├── data_preprocessing.py  # Scripts to convert videos to grayscale
│   ├── temporal_consistency.py # Temporal consistency (e.g., optical flow)
│   └── video_colorization.py   # Main script for video colorization
│
├── tests/                  # Automated tests for the project
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_colorization.py
│
├── tools/                  # Tools and utilities (e.g., for model explainability)
│   ├── explainability_tools.py
│
├── requirements.txt        # Development dependencies
├── setup.py                # Package and distribution management
└── README.md               # Project overview and setup instructions

## Contributors

### - Hayden Fargo
### - Tyler Lewis
### - Devyn Miller
### - Ponthea Zahraii
