# AI Text Recognition

This repository contains the code for a Handwritten Text Recognition (HTR) system using TensorFlow. The system includes training, validation, and inference processes, along with performance monitoring.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The AI Text Recognition system is designed to recognize handwritten text from images. It leverages deep learning models and provides tools for training, validating, and performing inference on text images. The project follows best practices for code version control, modularity, and documentation.

## Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow
- OpenCV
- NumPy
- Flask

### Clone the Repository

```bash
git clone https://github.com/maggierrr/AI-Text-Recognition.git
cd AI-Text-Recognition
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: Contains the model definition and setup.
- `func.py`: Contains the training function.
- `dataloader_iam.py`: Contains the data loading utilities (assumed to be part of the project).
- `README.md`: Project documentation.

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/htr-tensorflow.git
   cd htr-tensorflow
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, you need to prepare your dataset and specify the training parameters.

```bash
python src/train.py --data_dir data/raw --output_dir model/checkpoints
```

### Inference

To perform inference on an image:

```bash
python src/infer.py --model_dir model/checkpoints --image_path data/raw/test/image.png
```

### Running the Web App

You can also run a Flask web application to use the model via a web interface.

```bash
export FLASK_APP=src/app.py
flask run
```

## Project Structure

```
AI-Text-Recognition/
├── data/
│   ├── raw/
│   ├── processed/
│   └── ... # Data preparation scripts and files
├── model/
│   ├── __init__.py
│   ├── model.py
│   └── ... # Model definition and utilities
├── src/
│   ├── app.py
│   ├── train.py
│   ├── infer.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   └── ... # Source code for training, inference, and preprocessing
├── .github/ # GitHub workflows and actions
├── .gitignore
├── LICENSE.md
├── README.md
├── requirements.txt
└── ... # Additional files and scripts
```

## Model Definition

The model is defined in `model.py` and includes CNN, RNN, and CTC components for text recognition. It uses TensorFlow v1 compatibility mode with eager execution disabled.





## Model Training

### Data Preparation

Place your training and validation data in the `data/raw` directory. The data should be organized as follows:

```
data/
└── raw/
    ├── train/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    └── val/
        ├── image1.png
        ├── image2.png
        └── ...
```

### Training Command

```bash
python src/train.py --data_dir data/raw --output_dir model/checkpoints
```

### Training Parameters

You can customize training parameters using command-line arguments:

- `--epochs`: Number of training epochs.
- `--batch_size`: Size of each training batch.
- `--learning_rate`: Learning rate for the optimizer.

### Training Pipeline

1. Initialize variables and preprocess data.
2. Enter training loop:
   - Train on batches of data.
   - Perform validation.
   - Write summaries for monitoring.
   - Save the model if it improves.
   - Stop if early stopping criteria are met.

## Inference

To perform inference, use the provided `infer.py` script. You need to specify the path to the saved model and the image file.

### Inference Command

```bash
python src/infer.py --model_dir model/checkpoints --image_path data/raw/test/image.png
```

## Evaluation

The evaluation scripts allow you to measure the performance of your model using metrics such as Character Error Rate (CER) and Word Accuracy.

### Evaluation Command

```bash
python src/evaluate.py --model_dir model/checkpoints --data_dir data/raw/val
```

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

This README provides detailed instructions and explanations, covering all important aspects of the project. Feel free to adjust the specifics based on the actual content and structure of your project.