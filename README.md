# 🖐️ American Sign Language (ASL) Recognition Project

American Sign Language (ASL) is a visual language that is used by the deaf and hard-of-hearing community in the United States and some parts of Canada. American Sign Language is used as a means of communication for individuals who are deaf or hard-of-hearing. It enables them to express themselves and understand others in a natural, accessible way. It is also used by hearing individuals who work with the deaf and hard-of-hearing community, such as interpreters, teachers, and support staff. The use of ASL has been shown to have social, emotional, and cognitive benefits, and it is a valuable tool for promoting inclusivity and accessibility in society.

![ASL_Alphabet](project_images/ASL_Alphabet.png)
---

## 🎯 Main goals

My main goal is to build a machine learning model trained on thousands of images of American Sign Language alphabet signs that can generalize to instances of images of signs never seen by the model. This is because the ultimate goal of this model should be to create a useful model for various types of applications that try to predict the meaning of a letter sign starting from images captured in real-time, allowing us to communicate with deaf or hard-of-hearing individuals.

---

## 🤔 What will I try to do?

In this notebook, I explore different types of models to see which one fits better a sample image dataset for the ASL recognition problem taken from Kaggle. I compare the results obtained to choose the model with the best performance that I will later train on a dataset of **14448 ASL signs images built entirely by me**.  

Considering that we are talking about an image classification task, and since images are composed of pixels, we have a task in which the features of our dataset are the pixels of the images (these latter can be several hundreds or even thousands) leading the models to have very sparse parameters in the space, causing normal Machine Learning models to have poor performance in predicting the correct label of belonging.  

As a result, it is necessary to use more sophisticated **Deep Learning models**, particularly **neural networks** and **convolutional neural networks (CNNs)**. However, I will also implement an **ensemble model**, specifically a **Random Forest**. Since this is a complex problem where the features are many and the parameters are sparse in the space, **Clustering** turns out to be a very poor estimator, so I have avoided implementing it.

---

## 🛠️ What do I need?

**PySpark** is an open-source Python API for Apache Spark, a fast and general-purpose cluster computing system that is used for processing big data. PySpark provides a high-level interface for working with Spark and enables data scientists and engineers to leverage their existing Python skills to interact with Spark data structures and algorithms.  

PySpark is widely used for data analysis, machine learning, and other data-intensive tasks, making it a powerful tool for data engineers, data scientists, and developers.  

However, PySpark natively provides APIs to implement only machine learning models. It does not support Deep Learning models such as neural networks or convolutional neural networks. I had to search for frameworks that would also allow me to work with deep learning algorithms. After deep research of the best framework, I chose to use **Horovod**.

---

## 🚀 What is Horovod? Why is Horovod a good choice? What to install?

**Horovod** is an open source distributed deep learning framework that is designed to support **TensorFlow** and **PyTorch** models. It provides an easy-to-use, scalable and fast solution for distributed deep learning training on multiple GPUs/CPUs and nodes.  

Horovod is a good choice for image classification because it offers the following benefits:  

- ⚡ **Scalability**: Horovod allows you to scale your training across multiple GPUs/CPUs and nodes, making it possible to train large-scale models.  
- ⏩ **Speed**: Horovod can significantly speed up training times by distributing the workload among multiple GPUs/CPUs.  
- 🧩 **Easy to use**: Horovod is designed to be easy to use and integrate into existing deep learning pipelines.  
- 🔗 **Interoperability**: Horovod supports TensorFlow and PyTorch, making it possible to use a variety of models and techniques for image classification.  

Overall, Horovod provides a fast, scalable and easy-to-use solution for distributed deep learning training on multiple GPUs/CPUs, making it an ideal choice for image classification tasks.

---

## 📂 The ASL Dataset and preprocessing

The **Sign Language MNIST dataset** is a replacement for the popular MNIST image dataset of handwritten digits, aimed to provide a more challenging and original benchmark for computer vision and machine learning applications.  

It consists of hand gestures representing the 24 letters of the American Sign Language (excluding **J** and **Z** due to their motion-based gestures) in a format similar to the classic MNIST, with grayscale images of **28x28 pixels**.  

- 📊 **Dataset size**:  
  - Training data: **27,455 cases**  
  - Test data: **7,172 cases**  

Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter **A-Z** (excluding **J, Z**).  

The dataset format matches MNIST closely with:

label, pixel1, pixel2, …, pixel784

👉 For more information check the Kaggle dataset link.  

**Data preprocessing** involves:  
- 🧹 **Data cleansing** (removing invalid, incomplete, or irrelevant data)  
- 🔄 **Data transformation** (scaling, normalization, encoding)  
- 🔗 **Data integration** (combining multiple data sources)  
- ✂️ **Data reduction** (eliminating redundant/irrelevant data)  

The goal is to ensure the data is accurate, consistent, and ready for machine learning algorithms.

---

## ⚙️ Scaler, VectorAssembler, StringIndexer and OneHotEncoder for CNN

- **Scaling**: Pixel values are divided by **255** (max grayscale value) to normalize them between `0-1`.  
- **VectorAssembler**: Combines columns into a single vector column.  
- **StringIndexer**: Encodes string labels into numeric indices.  
- **OneHotEncoder**: Transforms category indices into binary vectors for CNN training.  

---

## 🧠 Convolutional Neural Network (CNN)

After preprocessing, the dataset is ready for modeling.  
The **Convolutional Neural Network (CNN)** is the main model, built with layers such as:  

- 🌀 **Convolutional** → learn local features  
- 🔽 **Max Pooling** → down-sample feature maps  
- 💧 **Dropout** → prevent overfitting  
- 🔗 **Dense** → final classification  

CNNs automatically learn relevant features and are highly effective for image classification tasks.  

---

## 🔧 Training strategy

In the following code cells, I show the training, validation, and testing of the best models selected after **hyperparameter tuning**.  

- I followed a **try and test strategy** because full **Cross-Validation** would consume too many resources (time and memory).  
- Even limited CV on CNN models caused **memory overflow**.  

So, the chosen strategy balanced performance tuning with available resources.

---
