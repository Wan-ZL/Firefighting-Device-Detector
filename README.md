# Firefighting Device Detection Project

This project is a Python (3.8.19) application that uses a modified SimCLR model for detecting firefighting devices in images. The model is trained on the 'Firefighting Device Detection Image Dataset' from Roboflow.

## Model Details

The model is a SimCLR but modified by replacing ResNet50 with a pre-trained ResNet 152 for higher performance. The performance of this model is test_acc@1: 84.9 and test_acc@5: 92.4 with model size 59.32M.

## How to Run

1. Clone the repository to your local machine.
2. Create a Conda environment with the specific Python version:
    ```
    conda create -n firefighting-dev python=3.8.19
    ```
3. Activate the newly created Conda environment:
    ```
    conda activate firefighting-dev
    ```
4. Install the required dependencies using pip:
    ```
    pip install -r requirements.txt
    ```
5. Run the main script:
    ```
    python main.py
    ```
6. You will be prompted to select if you want to use an existing pre-trained model or train a new one. Training a new model will take a lot of time, so it is suggested to use the pre-trained model for the first time.
7. The program will then ask if you want to test on a pre-selected symbol or use the mouse to crop the one you like. Try the pre-selected symbol first time to test if everything works properly.

## Dataset

The 'Firefighting Device Detection Image Dataset' from Roboflow is used in this project. The dataset is automatically downloaded and prepared for training when you run the script.

## Note

Training a new model can take a significant amount of time and computational resources. It is recommended to use the provided pre-trained model for initial testing.
