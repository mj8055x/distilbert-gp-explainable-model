# distilbert-gp

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GPyTorch](https://img.shields.io/badge/GPyTorch-blue.svg)](https://gpytorch.ai/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97Transformers-yellow.svg)](https://huggingface.co/transformers)

A `gpytorch` implementation of a **DistilBERT + Gaussian Process (GP)** model for text classification. It combines a transformer's feature extraction with a Bayesian GP layer for robust uncertainty estimates. Includes a reproducible experiment on SST-2 for end-to-end training and calibration.

## Overview

Standard transformer models like BERT or DistilBERT are highly effective at text classification, but they have two common drawbacks:
1.  They often produce **overconfident** predictions, even when wrong.
2.  They struggle to "know what they don't know," making it difficult to detect **out-of-distribution (OOD)** samples.

This project addresses these gaps by replacing the standard, simple classification head with a **Gaussian Process (GP)** layer. This hybrid approach gives us the best of both worlds:

* **Powerful Features:** We use a pre-trained DistilBERT as a deep feature extractor.
* **Principled Uncertainty:** We use a GP to perform Bayesian classification on top of these features.

By using a **Sparse Variational Gaussian Process (SVGP)**, this entire model can be trained **end-to-end** on a single GPU, jointly optimizing the DistilBERT weights and the GP hyperparameters.

## Key Features

* **End-to-End Training:** Jointly optimizes DistilBERT and the GP layer using a single variational objective (the ELBO).
* **Uncertainty Quantification:** The model outputs a full predictive distribution (mean and variance) for each prediction, not just a single probability.
* **Improved Calibration:** Produces probabilities that are more reflective of the true likelihood of correctness, measured via **Expected Calibration Error (ECE)**.
* **Out-of-Distribution (OOD) Detection:** The model's predictive variance can be used as a signal to identify inputs that are different from its training data.
* **Reproducible Experiment:** Includes a complete training and evaluation script for the **GLUE SST-2** dataset.


## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/distilbert-gp.git](https://github.com/YOUR_USERNAME/distilbert-gp.git)
    cd distilbert-gp
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    The dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

    **requirements.txt:**
    ```text
    torch>=1.13.0
    gpytorch>=1.9.0
    transformers>=4.25.0
    datasets>=2.10.0
    scikit-learn>=1.2.0
    numpy
    ```

## Usage

The `train.py` script contains the complete, reproducible experiment. It will automatically download the GLUE SST-2 dataset using the `datasets` library.

To run the training and evaluation:
```bash
python train.py
```

The script will:

Load the distilbert-base-uncased tokenizer and model.

Load and tokenize the SST-2 dataset.

Initialize the DistilBERTGPModel and BernoulliLikelihood.

Train the model for NUM_EPOCHS using the VariationalELBO loss.

After each epoch, it will evaluate the model on the validation set and report:

Accuracy

F1-Score

Expected Calibration Error (ECE)

Average Predictive Variance

Evaluation & Results
This model is designed to be evaluated on more than just accuracy.

Classification: Check Accuracy and F1 for performance on the task.

Calibration: A low ECE (e.g., < 0.05) indicates the model's confidence scores are reliable.

Uncertainty: The Average Variance provides a baseline for in-distribution data. To test OOD detection, you can run the trained model on a different dataset (e.g., IMDB reviews) and verify that the average variance is significantly higher.

Acknowledgements
1. Hugging Face for the transformers library and pre-trained models.
2. The GPyTorch Team for their flexible and powerful Gaussian Process library.
3. All concerned research contributors.
