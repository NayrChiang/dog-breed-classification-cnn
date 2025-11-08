# Dog Breed Classification with CNN

A comprehensive deep learning project developing and training multiple convolutional neural network (CNN) architectures for dog breed classification from images using PyTorch. This project demonstrates practical implementation of CNN architectures, data augmentation, regularization techniques, and hyperparameter optimization.

## Project Overview

**Course**: APS360 - Applied Machine Learning  
**Institution**: University of Toronto  
**Term**: Winter 2023

### Objective

Develop and train convolutional neural network models to classify dog breeds from images, comparing different architectures and optimization techniques to achieve the best performance.

## Dataset

**Stanford Dogs Dataset** with multiple configurations:
- **10 classes**: Afghan hound, bloodhound, whippet, briard, Rottweiler, miniature pinscher, Appenzeller, pug, Great Pyrenees, chow
- **40 classes configuration**
- **50 classes configuration**

## Model Architectures

### 1. 2-Layer CNN
- Custom 2-layer convolutional neural network
- Trained with batch size 128, learning rate 5e-4
- Tested on 10 and 50 class datasets

### 2. 3-Layer CNN
- Custom 3-layer architecture with batch normalization
- Implemented dropout and weight decay for regularization
- Trained on 40-class dataset with batch normalization layers
- Improved training stability and generalization

### 3. AlexNet
- Implemented AlexNet architecture
- Adapted for 10-class classification
- Used 227×227 input size (AlexNet standard)
- Batch size 128, learning rate 5e-4

## Technical Implementation

### Data Preprocessing
- Image resizing to 224×224 (CNN) and 227×227 (AlexNet)
- Dataset normalization using calculated mean and std values
- Train/validation/test split (80/10/10 or 70/15/15 configurations)

### Data Augmentation
- Random horizontal/vertical flipping
- Random rotation
- Random zooming
- Random shearing
- Random brightness adjustment

### Training Details
- **Framework**: PyTorch
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam with various learning rates (1e-3, 5e-4)
- **Regularization**:
  - Weight decay (0.001, 0.003)
  - Dropout layers
  - Batch normalization
- **Batch Sizes**: 64, 100, 128
- **Training Duration**: Up to 330 epochs (final model)

### Model Evaluation
- Implemented accuracy calculation functions
- Tracked training and validation accuracy over epochs
- Model checkpointing and saving
- Test set evaluation

## Performance Results

### Best Model Performance
- **Training Accuracy**: 66.75%
- **Validation Accuracy**: 56.10%
- **Test Accuracy**: 52.91%
- **Average Test Accuracy**: 49.42%

### Training Configuration
- **Final Model**: 3-layer CNN with batch normalization
- **Epochs**: 330
- **Batch Size**: 128
- **Learning Rate**: 5e-4
- **Weight Decay**: 0.003

## Key Achievements

- Successfully implemented and compared multiple CNN architectures (2-layer, 3-layer, AlexNet)
- Applied advanced techniques: batch normalization, dropout, weight decay
- Optimized hyperparameters (learning rate, batch size, weight decay)
- Achieved significant improvement through iterative model refinement
- Final model trained for 330 epochs with comprehensive evaluation
- Implemented robust data preprocessing and augmentation pipeline

## Skills Demonstrated

- **PyTorch Framework**: Proficiency in deep learning model development
- **CNN Architecture Design**: Understanding of convolutional layers, pooling, and fully connected layers
- **Deep Learning Training**: Model training, optimization, and evaluation
- **Hyperparameter Tuning**: Learning rate, batch size, weight decay optimization
- **Data Preprocessing**: Image normalization, resizing, and augmentation
- **Regularization Techniques**: Batch normalization, dropout, weight decay
- **Model Evaluation**: Accuracy calculation, validation strategies, checkpointing
- **Computer Vision Applications**: Image classification, multi-class classification

## Project Structure

```
dog-breed-classification-cnn/
├── 3-layer-baseline-cnn.ipynb      # Baseline 3-layer CNN implementation
├── 6-layer-final-cnn.ipynb        # Final optimized model
├── final-report.pdf                 # Project report
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- PyTorch
- NumPy
- Matplotlib
- PIL/Pillow

### Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd dog-breed-classification-cnn
```

2. Install required packages:
```bash
pip install torch torchvision numpy matplotlib pillow jupyter
```

3. Download the Stanford Dogs dataset:
   - Visit the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) website
   - Download and extract the dataset
   - Update the dataset path in the notebooks

## Usage

### Running the Notebooks

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open and run the notebooks:
   - `3-layer-baseline-cnn.ipynb`: Baseline model implementation
   - `6-layer-final-cnn.ipynb`: Final optimized model

### Training a Model

The notebooks contain complete training pipelines. Key steps:

1. **Load and preprocess data**:
   - Set dataset path
   - Apply data augmentation
   - Create data loaders

2. **Define model architecture**:
   - Choose architecture (2-layer, 3-layer, or AlexNet)
   - Configure layers and parameters

3. **Train the model**:
   - Set hyperparameters (learning rate, batch size, epochs)
   - Run training loop
   - Monitor training and validation accuracy

4. **Evaluate on test set**:
   - Load best model checkpoint
   - Evaluate on test set
   - Calculate final accuracy

### Example Training Configuration

```python
# Hyperparameters
batch_size = 128
learning_rate = 5e-4
weight_decay = 0.003
num_epochs = 330

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=weight_decay)

# Loss function
criterion = nn.CrossEntropyLoss()
```

## Model Architectures Details

### 2-Layer CNN
- Convolutional layers with ReLU activation
- Max pooling layers
- Fully connected layers
- Output layer with number of classes

### 3-Layer CNN (Final Model)
- Three convolutional blocks
- Batch normalization after each convolutional layer
- Dropout layers for regularization
- Fully connected layers with dropout
- Output layer

### AlexNet
- Standard AlexNet architecture
- 5 convolutional layers
- 3 fully connected layers
- Adapted for 10-class classification
- 227×227 input size

## Data Augmentation Pipeline

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomZoom(0.1),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[...], std=[...])
])
```

## Results Analysis

The final model achieved:
- **Training Accuracy**: 66.75% - Shows model's ability to learn from training data
- **Validation Accuracy**: 56.10% - Indicates generalization capability
- **Test Accuracy**: 52.91% - Final performance on unseen data
- **Average Test Accuracy**: 49.42% - Consistent performance across runs

The gap between training and validation accuracy suggests some overfitting, which could be addressed with:
- Additional regularization
- More data augmentation
- Early stopping
- Learning rate scheduling

## Future Improvements

- Implement transfer learning with pre-trained models (ResNet, VGG, etc.)
- Experiment with different optimizers (SGD with momentum, AdamW)
- Implement learning rate scheduling
- Add more aggressive data augmentation
- Try ensemble methods
- Implement early stopping to prevent overfitting

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- APS360 course instructors and TAs
- University of Toronto for providing resources and facilities
- Stanford Vision Lab for the Stanford Dogs Dataset

## References

- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- AlexNet: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks.
