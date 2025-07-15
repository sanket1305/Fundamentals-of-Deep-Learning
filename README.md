## 🧠 Fundamentals of Deep Learning

This repository contains a curated collection of hands-on labs completed as part of the NVIDIA Deep Learning Institute (DLI) course: Fundamentals of Deep Learning. The labs provide practical experience in building and training deep neural networks using real-world computer vision and natural language processing (NLP) tasks.

---

### 🖊️ Lab 1: Handwritten Digit Recognition

#### Objective:
This lab introduces the foundational concepts of deep learning through a classic image classification task. You’ll explore the limitations of traditional programming approaches and understand how neural networks can solve pattern recognition problems effectively.

#### Key Concepts Covered:
- Introduction to the MNIST dataset
- Data loading and preprocessing using torchvision
- Designing and training a simple feedforward neural network with PyTorch
- Evaluating model performance on handwritten digits

#### Dataset Used:
MNIST (70,000 grayscale images of handwritten digits, 28x28 pixels)

#### Frameworks & Tools:
PyTorch, Torchvision

#### Key Learnings:
This exercise demonstrated how to build a basic neural network for digit classification using PyTorch. By training the model on MNIST, it achieved high accuracy on unseen data, showcasing the capability of deep learning models to generalize from relatively small and structured datasets. It also reinforced the importance of proper data preparation and model architecture in achieving good performance.

---

### 🤟 Lab 2: Image Classification of American Sign Language (ASL)

#### Objective:
This lab focuses on developing an image classification model to recognize American Sign Language (ASL) gestures. It covers the full model development pipeline—from data preprocessing to model evaluation—using a real-world image dataset.

#### Key Concepts Covered:
- Image preprocessing and data exploration using pandas and matplotlib
- Building and compiling a Neural Network model in PyTorch
- Training the model on labeled ASL gesture images
- Analyzing training vs. validation accuracy to assess overfitting

#### Dataset Used:
ASL Alphabet Dataset from Kaggle

#### Frameworks & Tools:
PyTorch, Matplotlib, Pandas

#### Key Learnings:
The model achieved high training accuracy but comparatively lower validation accuracy, indicating potential overfitting. This lab highlighted the importance of generalization and the need for techniques such as regularization or data augmentation to improve validation performance on more complex, real-world datasets.

---

### 🧠 Lab 3: Image Classification of ASL using Convolutional Neural Networks (CNNs)

#### Objective:
This lab advances the previous ASL image classification task by introducing Convolutional Neural Networks (CNNs)—a specialized architecture for image-based tasks. The exercise focuses on constructing deeper networks with enhanced feature extraction capabilities.

#### Key Concepts Covered:
- CNN-specific data preparation
- Designing a CNN with multiple convolutional, pooling, and fully connected layers
- Training and evaluating the CNN model on the ASL dataset
- Interpreting training stability and generalization from accuracy trends

#### Dataset Used:
ASL Alphabet Dataset from Kaggle

#### Frameworks & Tools:
PyTorch, Pandas

#### Key Learnings:
Replacing the basic neural network with a CNN significantly boosted both training and validation accuracy, demonstrating the power of convolutional architectures for visual pattern recognition. However, fluctuations in validation accuracy hinted at generalization issues, emphasizing the need for further improvements such as more data, regularization, or augmentation to enhance model robustness.

---

### 🧪 Lab 4A: Data Augmentation of ASL Images

#### Objective:
This lab introduces data augmentation techniques to enhance the ASL dataset, thereby improving model generalization. It also covers saving trained models for future deployment.

#### Key Concepts Covered:
- Applying real-time image transformations using torchvision.transforms
- Retraining the CNN model on augmented image data
- Observing improvements in validation stability and reduction in overfitting
- Saving model checkpoints to disk for inference use

#### Dataset Used:
ASL Alphabet Dataset from Kaggle

#### Frameworks & Tools:
PyTorch, Torchvision, Pandas, Matplotlib

#### Key Learnings:
Data augmentation proved effective in improving validation accuracy and reducing overfitting. While the training accuracy slightly decreased, the model demonstrated stronger generalization by being exposed to more diverse visual patterns. The resulting model was more robust and suitable for deployment scenarios.

### 🚀 Lab 4B: Deploying a Model Trained on Augmented ASL Images

#### Objective:
This lab demonstrates how to deploy a pre-trained model for inference. It focuses on loading saved weights, adapting input formats, and evaluating the model on unseen images.

#### Key Concepts Covered:
- Loading a trained PyTorch model from disk
- Preprocessing images with different formats (color, resolution) for inference
- Performing real-time predictions on unfamiliar ASL images
- Assessing model accuracy on diverse input data

#### Dataset Used:
ASL Alphabet Dataset from Kaggle (with custom test images)

#### Frameworks & Tools:
PyTorch, Torchvision, Pandas, Matplotlib

#### Key Learnings:
The deployed model successfully predicted ASL letters from new, high-resolution, and colored images—showcasing its ability to generalize beyond the original training distribution. This validated the effectiveness of both model architecture and the augmentation strategies used in prior stages.

--- 

