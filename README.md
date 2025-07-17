# 🧠 Fundamentals of Deep Learning

This repository contains a curated collection of hands-on labs completed as part of the **NVIDIA Deep Learning Institute (DLI)** course: *Fundamentals of Deep Learning*. These labs provide practical experience in designing, training, and deploying deep learning models for both computer vision and natural language processing (NLP) tasks.

## ⚙️ Technologies Used
- Python
- PyTorch
- Torchvision
- Pandas, Matplotlib
- Hugging Face Transformers
- PIL, glob

## 🧪 Hands-On Exercises

### 🖊️ Lab 1: Handwritten Digit Recognition

- **Dataset**: MNIST
- **Goal**: Build and train a simple neural network using PyTorch
- **Key Skills**: Data loading, feedforward networks, evaluation
- **Outcome**: High accuracy on digit classification, foundational DL pipeline understanding

### 🤟 Lab 2: Image Classification of ASL

- **Dataset**: ASL Alphabet from Kaggle
- **Goal**: Train a basic image classifier
- **Key Skills**: CNN introduction, image preprocessing
- **Outcome**: Identified overfitting and the need for improved generalization

### 🧠 Lab 3: ASL Classification Using CNNs

- **Dataset**: ASL Alphabet from Kaggle
- **Goal**: Replace basic NN with CNN for better performance
- **Key Skills**: Convolutional layers, pooling, model depth
- **Outcome**: Improved accuracy; validation instability highlighted model tuning needs

### 🧪 Lab 4A: Data Augmentation of ASL Images

- **Dataset**: ASL Alphabet from Kaggle
- **Goal**: Apply augmentation to improve generalization
- **Key Skills**: torchvision.transforms, model retraining
- **Outcome**: Increased validation accuracy and robustness

### 🚀 Lab 4B: Deploying a Model Trained on Augmented ASL Images

- **Dataset**: Custom image folder
- **Goal**: Load model and perform inference on unseen data
- **Key Skill**s: Model loading, image formatting, inference
- **Outcome**: Accurate predictions on colored, high-resolution images

### 🐶 Lab 5A: Automated Doggy Door Using Pretrained Model

- **Dataset**: Custom image folder
- **Goal**: Use pretrained model for pet image recognition
- **Key Skills**: Transfer learning, Torchvision models
- **Outcome**: Functional classification system with minimal code

### 🐕 Lab 5B: Presidential Doggy Door Using Transfer Learning

- **Dataset**: Small custom dataset
- **Goal**: Fine-tune a pretrained model with minimal data
- **Key Skills**: Layer freezing, model fine-tuning
- **Outcome**: Achieved high accuracy with a small dataset, showcasing the power of transfer learning

### 🗣️ Lab 6: Natural Language Processing with Transformers

- **Dataset**: Sample sentences
- **Goal**: Perform QA using BERT-style transformer
- **Key Skills**: Tokenization, embeddings, transformer inference
- **Outcome**: Used an LLM to accurately extract answers from text passages

### 🏁 Final Assessment: Fresh vs. Rotten Fruit Classifier

- **Objective**: Build a model that classifies fresh vs. rotten fruit with ≥92% validation accuracy
- **Tools Used**: PyTorch, Torchvision, PIL, Glob
- **Techniques Applied**: Transfer learning, data augmentation, fine-tuning
- **Result**: ✅ Achieved 97.26% validation accuracy
- **Outcome**: Validated ability to build a production-ready classifier using limited data and pretrained models

### 🎓 Key Learning Outcomes

- Designed, trained, and evaluated deep learning models using PyTorch
- Applied convolutional neural networks (CNNs) for image-based tasks
- Leveraged transfer learning and pretrained models for low-data scenarios
- Implemented data augmentation to reduce overfitting and improve generalization
- Used transformer-based models for natural language processing
- Deployed trained models for real-world inference tasks

## 📜 License
This repository is for educational purposes as part of the NVIDIA DLI course. Please consult NVIDIA DLI Terms of Use for more information.

## 🙌 Acknowledgements
Special thanks to NVIDIA Deep Learning Institute for providing the content, tools, and cloud infrastructure to complete this hands-on learning experience.

