# 5C Brain MRI Metastasis Segmentation Assignment

Brain MRI Metastasis Segmentation using Machine Learning and Streamlit
This project focuses on developing a machine learning-based solution for segmenting metastatic tumors in brain MRI scans. The segmentation process aims to automatically detect and isolate metastatic regions in MRI images, aiding in early diagnosis and treatment planning for patients with cancer that has spread to the brain.

## Key Features:
Dataset: The project uses a brain MRI dataset with labeled regions of metastasis, enabling supervised learning. The dataset includes multiple MRI modalities, such as T1, T2, and FLAIR images, for a more comprehensive analysis.
Preprocessing: MRI scans are preprocessed to enhance the image quality and standardize input dimensions. This includes techniques like normalization, skull stripping, and augmentation to increase the robustness of the model.
Modeling: A machine learning pipeline is built to perform the segmentation task. Popular models like U-Net, or other deep learning architectures designed for medical image segmentation, are trained using the processed MRI scans.
Evaluation: The performance of the model is evaluated using metrics such as Dice coefficient, Intersection over Union (IoU), and precision-recall to assess the accuracy of tumor segmentation.
## Streamlit Application:
The project is deployed using Streamlit, providing an interactive web interface that allows users to upload MRI images and visualize segmentation results in real-time. The app includes:

Image Uploading: Users can upload their own MRI scans for analysis.
Segmentation Output: The segmented regions of metastasis are highlighted on the uploaded brain MRI scans.
Model Metrics: A section displaying the performance metrics of the model, showcasing accuracy and confidence in the predictions.
## Objective:
The primary objective of this project is to create an easy-to-use, automated system for segmenting brain metastases from MRI scans, leveraging the power of deep learning and presenting the solution through a user-friendly web interface.

## Tools and Libraries:
Machine Learning: TensorFlow/Keras, PyTorch (for model development)
Medical Imaging Libraries: nibabel, SimpleITK (for MRI data handling)
Web Interface: Streamlit (for deployment and user interaction)
Others: NumPy, OpenCV, Matplotlib (for data preprocessing and visualization)
This project has the potential to contribute significantly to healthcare by improving the efficiency of brain tumor diagnosis and providing support to medical professionals.


## Requirement:
    keras==2.7.0
    matplotlib==3.3.4
    numpy==1.20.1
    opencv_python==4.5.4.60
    Pillow==9.0.1
    scikit_learn==1.0.2
    tensorflow==2.7.0
