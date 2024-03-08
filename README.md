# 🌆 Cityscapes using U-Net 🖼️

## Description
The Cityscapes using U-Net project is an image segmentation task aimed at segmenting urban street scenes from the Cityscapes dataset using the U-Net architecture. U-Net is a convolutional neural network (CNN) architecture designed for biomedical image segmentation, but it has proven effective for various segmentation tasks, including urban scene segmentation.

## Overview
This project utilizes the Cityscapes dataset, which contains high-quality images with pixel-level annotations for various urban street scenes. The U-Net architecture is employed to learn a mapping from input images to pixel-level segmentation masks, effectively separating different semantic classes such as roads, vehicles, pedestrians, and buildings.

## Project Structure 📂
- `data/`: Contains the Cityscapes dataset and preprocessed data.
- `model/`: Stores the U-Net model architecture and trained weights.
- `notebooks/`: Jupyter Notebooks for data preprocessing, model training, and evaluation.
- `README.md`: Overview of the project and instructions for usage.

## Installation 🛠️
1. Clone the repository to your local machine.
2. Install the required libraries using the following command:

```bash
pip install torch
pip install torchvision
pip install pandas
pip install scikit-learn
pip install matplotlib
```

3. Download the Cityscapes dataset and place it in the `data/` directory.

## Usage 🚀
1. Preprocess the Cityscapes dataset using the provided Jupyter Notebooks in the `notebooks/` directory.
2. Train the U-Net model on the preprocessed data using the training notebook.
3. Evaluate the trained model's performance on the test set using the evaluation notebook.
4. Use the trained model for inference on new urban street scene images.

## Model Evaluation 📊
The model's performance is evaluated using metrics such as Intersection over Union (IoU), Pixel Accuracy, and Mean Intersection over Union (mIoU) on the test set.

## Results and Visualizations 📈
Visualizations of segmentation masks, along with accuracy and loss curves during training, are provided in the evaluation notebook to illustrate the model's performance.

![Image 2](https://github.com/kashifmehdi/City-Scapes/blob/26a5208b4999089ab76f565f7f648dc37de24948/Result.png)

## Contributing 🤝
Contributions to the project are welcome! If you'd like to contribute enhancements or additional features, feel free to fork the repository and submit a pull request.

## Acknowledgments 🙏
- The Cityscapes dataset used in this project is provided by the Cityscapes Dataset Team.
- Special thanks to the authors of the U-Net architecture for their groundbreaking work in image segmentation.

## References 📚
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
