# Optimizers using First Principle Approach

This mini project is focused on understanding how optimization algorithms—particularly gradient descent and its variants—work by implementing them from scratch. The primary goal is to gain a deep insight into the mechanics of parameter updates, momentum, and their overall impact on loss and accuracy during training.

---

## Table of Contents

- [Introduction](#introduction)  
- [Project Overview](#project-overview)  
- [Implementation Details](#implementation-details)  
  - [Custom Optimizer Implementation](#custom-optimizer-implementation)  
  - [Keras Optimizer Implementation](#keras-optimizer-implementation)  
- [Comparative Analysis: Loss and Accuracy](#comparative-analysis-loss-and-accuracy)  
- [Usage](#usage)  
- [Conclusion](#conclusion)  

---

## Introduction

The main agenda of this project is to understand the working of the gradient descent algorithm and related optimization techniques by implementing them from scratch. Rather than focusing solely on a specific model, the emphasis here is on the optimizer itself. This project demonstrates how a custom optimizer (using gradient descent with momentum) can be built and compared with high-level optimizers provided by frameworks such as Keras.

---

## Project Overview

### **Objective**  
To explore the inner workings of optimizers by:  
- Implementing gradient descent with momentum from scratch.  
- Observing the effects of hyperparameters such as learning rate, batch size, and regularization on loss and accuracy.  
- Comparing the custom optimizer’s performance with that of Keras’s built-in optimizers (e.g., SGD).  

### **Focus**  
The focus is on the optimization process: how gradients are computed, how parameters are updated, and how these updates influence model performance in terms of loss and accuracy.  

---

## Implementation Details

### **Custom Optimizer Implementation**

#### **Gradient Computation**  
The gradients of the loss function (binary cross-entropy with L2 regularization) with respect to model parameters are computed manually.

```python
def compute_gradients(X, y, w, b, lambda_reg=0.001):
    m = X.shape[0]
    z = X.dot(w) + b
    A = sigmoid(z)
    dz = A - y
    dw = (1/m) * X.T.dot(dz) + (lambda_reg/m) * w  
    db = (1/m) * np.sum(dz)
    return dw, db
```

#### **Momentum Update and Training Loop**  
Parameters are updated using gradient descent with momentum.

```python
learning_rate = 0.005
num_epochs = 100
batch_size = 32
lambda_reg = 0.001

beta = 0.9
velocity_w = np.zeros_like(w)
velocity_b = np.zeros_like(b)

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(1, num_epochs + 1):
    for i in range(0, m, batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        dw, db = compute_gradients(X_batch, y_batch, w, b, lambda_reg)

        # Apply Momentum Update
        velocity_w = beta * velocity_w + (1 - beta) * dw
        velocity_b = beta * velocity_b + (1 - beta) * db
        w -= learning_rate * velocity_w
        b -= learning_rate * velocity_b

    # Compute loss and accuracy
    train_loss = compute_loss(X_train, y_train, w, b, lambda_reg)
    val_loss = compute_loss(X_val, y_val, w, b, lambda_reg)
    train_acc = compute_accuracy(X_train, y_train, w, b)
    val_acc = compute_accuracy(X_val, y_val, w, b)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch}/{num_epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
```

#### **Activation Function**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

#### **Loss Function with Regularization**
```python
def compute_loss(X, y, w, b, lambda_reg=0.001):
    m = X.shape[0]
    z = X.dot(w) + b
    A = sigmoid(z)
    loss = - (1/m) * np.sum(y * np.log(A + 1e-8) + (1 - y) * np.log(1 - A + 1e-8))
    reg_term = (lambda_reg / (2 * m)) * np.sum(w**2)
    return loss + reg_term
```

#### **Accuracy Calculation**
```python
def compute_accuracy(X, y, w, b):
    z = X.dot(w) + b
    A = sigmoid(z)
    preds = A > 0.5
    return np.mean(preds == y)
```

---

### **Keras Optimizer Implementation**

#### **Model Building**
```python
model = Sequential([
    Dense(1, input_shape=(X_train.shape[1],), activation='sigmoid')
])
```

#### **Compilation**
```python
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

#### **Training**
```python
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=32)
```

---

## Comparative Analysis: Loss and Accuracy

| Optimizer        | Loss Reduction | Accuracy Improvement |
|-----------------|---------------|--------------------|
| **Custom Optimizer** | Gradually decreases, requires tuning | Improves but may fluctuate |
| **Keras SGD** | Faster convergence, lower loss | More stable accuracy |

---

## Usage

### **Clone the Repository**
```bash
git clone https://github.com/yourusername/optimizers-from-scratch.git
cd optimizers-from-scratch
```

### **Set Up a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Run the Jupyter Notebook**
```bash
jupyter notebook
```

---

## Conclusion

This project enhances the understanding of optimization algorithms, demonstrating the impact of hyperparameters and momentum on training dynamics. While custom implementations provide deep insights, high-level optimizers like Keras’s SGD are more efficient for practical use.

