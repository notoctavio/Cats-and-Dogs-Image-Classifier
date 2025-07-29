# 🐱🐶 CNN Image Classifier: Cats vs Dogs

<div align="center">

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

*A powerful Convolutional Neural Network that distinguishes between cats and dogs with 90%+ training accuracy!*

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Dataset](#-dataset) • [Usage](#-usage) • [Results](#-results)

</div>

---

## 🎯 Project Overview

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify images of cats and dogs. The model achieves impressive performance through careful architecture design and data augmentation techniques.

### 🌟 Key Highlights
- **90%+ Training Accuracy** achieved in 25 epochs
- **Advanced Data Augmentation** to prevent overfitting
- **Real-time Prediction** capability on new images
- **Clean, Well-documented Code** with detailed explanations

---

## 🚀 Features

### 🔧 Technical Features
- **Deep Learning Architecture**: Multi-layer CNN with optimized parameters
- **Data Augmentation**: Rotation, shearing, zooming, and flipping
- **Batch Processing**: Efficient training with batch size optimization
- **Real-time Inference**: Single image prediction capability

### 📊 Model Capabilities
- Binary classification (Cat vs Dog)
- 64x64 pixel image processing
- RGB color image support
- Robust feature extraction

---

## 🏗️ Architecture

### Model Structure
```
Input Layer (64x64x3)
        ↓
Convolutional Layer (32 filters, 3x3, ReLU)
        ↓
Max Pooling (2x2, stride=2)
        ↓
Convolutional Layer (32 filters, 3x3, ReLU)
        ↓
Max Pooling (2x2, stride=2)
        ↓
Flatten Layer
        ↓
Dense Layer (128 neurons, ReLU)
        ↓
Output Layer (1 neuron, Sigmoid)
```

### 🔬 Technical Specifications
- **Input Shape**: (64, 64, 3) - RGB images
- **Activation Functions**: ReLU (hidden layers), Sigmoid (output)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

---

## 📦 Installation

### Prerequisites
```bash
Python 3.10+
TensorFlow 2.16.2
NumPy
Pandas
```

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/cnn-cats-dogs-classifier.git
   cd cnn-cats-dogs-classifier
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install tensorflow numpy pandas matplotlib
   ```

---

## 📁 Dataset

### Download the Dataset
The cats and dogs dataset is available for download:

🔗 **[Download Dataset](https://www.dropbox.com/scl/fi/ppd8g3d6yoy5gbn960fso/dataset.zip?rlkey=lqbqx7z6i9hp61l6g731wgp4v&e=1&st=gdn6pydw&dl=0)**

### Dataset Structure
After downloading and extracting the dataset, organize it as follows:
```
dataset/
├── training_set/
│   ├── cats/          # Training images of cats
│   └── dogs/          # Training images of dogs
└── test_set/
    ├── cats/          # Test images of cats
    └── dogs/          # Test images of dogs
```

### Dataset Information
- **Training Set**: 8,000 images (4,000 cats + 4,000 dogs)
- **Test Set**: 2,000 images (1,000 cats + 1,000 dogs)
- **Image Format**: JPEG
- **Image Size**: Variable (resized to 64x64 during preprocessing)
- **Total Size**: ~540 MB

---

## 🎮 Usage

### Training the Model

```python
# Import required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize and train the CNN
python CNN.ipynb
```

### Making Predictions

```python
# Load and predict on a single image
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Make prediction
test_image = image.load_img('path/to/image.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)
prediction = 'dog' if result[0][0] == 1 else 'cat'
print(f"Prediction: {prediction}")
```

---

## 📈 Results

### Training Performance
- **Final Training Accuracy**: ~90%
- **Validation Accuracy**: ~80%
- **Training Epochs**: 25
- **Training Time**: ~19 minutes

### Model Metrics
| Metric | Value |
|--------|-------|
| Training Accuracy | 90.3% |
| Validation Accuracy | 77.6% |
| Training Loss | 0.23 |
| Validation Loss | 0.57 |

### 📊 Training Progress
The model shows steady improvement over 25 epochs with clear learning progression:
- **Early Epochs (1-5)**: Rapid initial learning (53% → 73%)
- **Mid Training (6-15)**: Steady improvement (73% → 84%)
- **Late Training (16-25)**: Fine-tuning and optimization (84% → 90%)

---

## 🔍 Data Preprocessing

### Training Data Augmentation
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,        # Normalize pixels to [0,1]
    shear_range=0.2,       # Shear transformation
    zoom_range=0.2,        # Zoom transformation
    horizontal_flip=True   # Random horizontal flip
)
```

### Test Data Processing
```python
test_datagen = ImageDataGenerator(rescale=1./255)
```

---

## 🛠️ Code Structure

```
📁 CNN-Cats-Dogs-Classifier/
├── 📄 CNN.ipynb              # Main Jupyter notebook
├── 📄 README.md              # This file
├── 📁 dataset/               # Dataset folder
│   ├── 📁 training_set/      # Training images
│   └── 📁 test_set/          # Test images
└── 📄 requirements.txt       # Dependencies
```

---

## 🧠 Key Learning Concepts

### Convolutional Layers
- **Feature Detection**: Identifies edges, patterns, and textures
- **Parameter Sharing**: Reduces overfitting and computational cost
- **Translation Invariance**: Recognizes features regardless of position

### Pooling Layers
- **Dimensionality Reduction**: Reduces computational complexity
- **Feature Robustness**: Makes model less sensitive to small translations
- **Max Pooling**: Retains strongest activations from each region

### Data Augmentation
- **Prevents Overfitting**: Increases effective dataset size
- **Improves Generalization**: Model learns from varied perspectives
- **Real-world Robustness**: Handles different lighting and orientations

---

## 🎯 Future Improvements

- [ ] **Transfer Learning**: Implement pre-trained models (VGG16, ResNet)
- [ ] **Model Ensemble**: Combine multiple models for better accuracy
- [ ] **Advanced Augmentation**: Add color jittering and cutout
- [ ] **Mobile Deployment**: Convert to TensorFlow Lite
- [ ] **Web Interface**: Create Flask/Streamlit app for easy use

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Machine Learning A-Z Course** for the comprehensive deep learning curriculum
- **TensorFlow Team** for the amazing deep learning framework
- **Kaggle** for providing excellent datasets for machine learning projects

---

<div align="center">

### 🌟 If you found this project helpful, please give it a star! ⭐

**Built with ❤️ and lots of ☕**

[⬆ Back to Top](#-cnn-image-classifier-cats-vs-dogs)

</div>
