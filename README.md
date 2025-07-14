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