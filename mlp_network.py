"""
5-Layer Image Classification Neural Network implemented with NumPy
Architecture: Flatten -> Dense -> ReLU -> Dense -> ReLU -> Dense -> ReLU -> Dense -> Softmax

Designed for image classification tasks (e.g., MNIST, CIFAR-10)
"""

import numpy as np
from math import inf
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image


# ===================== Model Initialization =====================

def initialize_model(input_size=784, hidden1=1024, hidden2=512, hidden3=256, hidden4=128, hidden5=64, num_classes=10):
    """
    Initialize a 6-layer image classification neural network (5 hidden layers).
    
    Layer structure:
    1. Input Layer (Flatten): input_size
    2. Dense Layer 1 + ReLU: input_size -> hidden1
    3. Dense Layer 2 + ReLU: hidden1 -> hidden2
    4. Dense Layer 3 + ReLU: hidden2 -> hidden3
    5. Dense Layer 4 + ReLU: hidden3 -> hidden4
    6. Dense Layer 5 + ReLU: hidden4 -> hidden5
    7. Output Dense Layer + Softmax: hidden5 -> num_classes
    """
    np.random.seed(42)
    
    model = {
        # Layer 1: Dense (input_size -> hidden1)
        'W1': np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size),
        'b1': np.zeros((1, hidden1)),
        
        # Layer 2: Dense (hidden1 -> hidden2)
        'W2': np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1),
        'b2': np.zeros((1, hidden2)),
        
        # Layer 3: Dense (hidden2 -> hidden3)
        'W3': np.random.randn(hidden2, hidden3) * np.sqrt(2.0 / hidden2),
        'b3': np.zeros((1, hidden3)),
        
        # Layer 4: Dense (hidden3 -> hidden4)
        'W4': np.random.randn(hidden3, hidden4) * np.sqrt(2.0 / hidden3),
        'b4': np.zeros((1, hidden4)),
        
        # Layer 5: Dense (hidden4 -> hidden5)
        'W5': np.random.randn(hidden4, hidden5) * np.sqrt(2.0 / hidden4),
        'b5': np.zeros((1, hidden5)),
        
        # Layer 6: Output Dense (hidden5 -> num_classes)
        'W6': np.random.randn(hidden5, num_classes) * np.sqrt(2.0 / hidden5),
        'b6': np.zeros((1, num_classes)),
        
        # Store config
        'config': {
            'input_size': input_size,
            'hidden1': hidden1,
            'hidden2': hidden2,
            'hidden3': hidden3,
            'hidden4': hidden4,
            'hidden5': hidden5,
            'num_classes': num_classes
        }
    }
    
    return model


# ===================== Activation Functions =====================

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)


def softmax(x):
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ===================== Forward Pass =====================

def forward(images, model):
    """
    Forward pass through the 6-layer image classification network.
    """
    cache = {}
    
    # Flatten input
    if len(images.shape) > 2:
        batch_size = images.shape[0]
        flattened = images.reshape(batch_size, -1)
    else:
        flattened = images
    
    cache['input'] = flattened
    
    # Layer 1: Dense + ReLU
    z1 = np.dot(flattened, model['W1']) + model['b1']
    a1 = relu(z1)
    cache['z1'], cache['a1'] = z1, a1
    
    # Layer 2: Dense + ReLU
    z2 = np.dot(a1, model['W2']) + model['b2']
    a2 = relu(z2)
    cache['z2'], cache['a2'] = z2, a2
    
    # Layer 3: Dense + ReLU
    z3 = np.dot(a2, model['W3']) + model['b3']
    a3 = relu(z3)
    cache['z3'], cache['a3'] = z3, a3

    # Layer 4: Dense + ReLU
    z4 = np.dot(a3, model['W4']) + model['b4']
    a4 = relu(z4)
    cache['z4'], cache['a4'] = z4, a4

    # Layer 5: Dense + ReLU
    z5 = np.dot(a4, model['W5']) + model['b5']
    a5 = relu(z5)
    cache['z5'], cache['a5'] = z5, a5
    
    # Layer 6: Output layer + Softmax
    z6 = np.dot(a5, model['W6']) + model['b6']
    output = softmax(z6)
    cache['z6'] = z6
    cache['output'] = output
    
    return output, cache


# ===================== Loss Function =====================

def cross_entropy_loss(output, labels):
    """
    Compute cross-entropy loss.
    
    Parameters:
    -----------
    output : np.array
        Model predictions (batch_size, num_classes)
    labels : np.array
        Ground truth labels (batch_size,) - class indices
    
    Returns:
    --------
    loss : float
        Average cross-entropy loss
    """
    batch_size = output.shape[0]
    output_clipped = np.clip(output, 1e-7, 1 - 1e-7)
    correct_probs = output_clipped[np.arange(batch_size), labels]
    loss = -np.mean(np.log(correct_probs))
    return loss


# ===================== Backward Pass =====================

def backward(output, labels, cache, model):
    """
    Backward pass - compute gradients for the 6-layer network.
    """
    batch_size = output.shape[0]
    gradients = {}
    
    # Layer 6: Output gradient
    dz6 = output.copy()
    dz6[np.arange(batch_size), labels] -= 1
    dz6 /= batch_size
    
    gradients['W6'] = np.dot(cache['a5'].T, dz6)
    gradients['b6'] = np.sum(dz6, axis=0, keepdims=True)
    
    # Layer 5 backprop
    da5 = np.dot(dz6, model['W6'].T)
    dz5 = da5 * relu_derivative(cache['z5'])
    gradients['W5'] = np.dot(cache['a4'].T, dz5)
    gradients['b5'] = np.sum(dz5, axis=0, keepdims=True)

    # Layer 4 backprop
    da4 = np.dot(dz5, model['W5'].T)
    dz4 = da4 * relu_derivative(cache['z4'])
    gradients['W4'] = np.dot(cache['a3'].T, dz4)
    gradients['b4'] = np.sum(dz4, axis=0, keepdims=True)
    
    # Layer 3 backprop
    da3 = np.dot(dz4, model['W4'].T)
    dz3 = da3 * relu_derivative(cache['z3'])
    gradients['W3'] = np.dot(cache['a2'].T, dz3)
    gradients['b3'] = np.sum(dz3, axis=0, keepdims=True)
    
    # Layer 2 backprop
    da2 = np.dot(dz3, model['W3'].T)
    dz2 = da2 * relu_derivative(cache['z2'])
    gradients['W2'] = np.dot(cache['a1'].T, dz2)
    gradients['b2'] = np.sum(dz2, axis=0, keepdims=True)
    
    # Layer 1 backprop
    da1 = np.dot(dz2, model['W2'].T)
    dz1 = da1 * relu_derivative(cache['z1'])
    gradients['W1'] = np.dot(cache['input'].T, dz1)
    gradients['b1'] = np.sum(dz1, axis=0, keepdims=True)
    
    return gradients


# ===================== Parameter Update =====================

def update_parameters(model, gradients, learning_rate=0.01):
    """
    Update model parameters using gradient descent.
    """
    params = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4', 'W5', 'b5', 'W6', 'b6']
    for key in params:
        if key in gradients:
            model[key] = model[key] - learning_rate * gradients[key]
    return model


# ===================== Dataset =====================

# ===================== Dataset =====================

def load_dataset(root_path, image_size=64):
    """
    Load real images from the classification dataset directory.
    
    Structure: root_path/class_name/image.jpg
    
    Returns:
    --------
    data : list of tuples
        List of (flattened_image, label_index) pairs
    class_names : list
        List of class names ordered by index
    """
    if not os.path.exists(root_path):
        print(f"Error: Path {root_path} does not exist.")
        return [], []

    class_names = sorted(os.listdir(root_path))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    data = []
    print(f"Loading dataset from {root_path}...")
    
    for class_name in class_names:
        class_dir = os.path.join(root_path, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            img_path = os.path.join(class_dir, f)
            try:
                # Load image, convert to grayscale, resize
                with Image.open(img_path) as img:
                    img = img.convert('L') # Grayscale to keep input size small
                    img = img.resize((image_size, image_size))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    
                    # Flatten image
                    flattened = img_array.flatten()
                    
                    label = class_to_idx[class_name]
                    data.append((flattened, label))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    return data, class_names


# ===================== Save/Load Model =====================

def save_model(model, path='best_model.npz'):
    """Save model to file"""
    np.savez(path, **model)
    print(f"Model saved to {path}")


def load_model(path='best_model.npz'):
    """Load model from file"""
    data = np.load(path, allow_pickle=True)
    model = {key: data[key] for key in data.files}
    if isinstance(model.get('config'), np.ndarray):
        model['config'] = model['config'].item()
    print(f"Model loaded from {path}")
    return model


# ===================== Visualization =====================

def draw_loss(history_path='loss_history.csv', save_path='loss_plot.png'):
    """
    Read loss history from CSV and plot train/validation loss.
    """
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found.")
        return

    df = pd.read_csv(history_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")


# ===================== Training Loop =====================

def train(model, train_path, val_path, image_size=64, epochs=10, learning_rate=0.1, batch_size=32, csv_path='loss_history.csv'):
    """
    Train the model and save loss history to CSV.
    """
    train_dataset, class_names = load_dataset(train_path, image_size=image_size)
    val_dataset, _ = load_dataset(val_path, image_size=image_size)
    
    if not train_dataset:
        print("Empty training dataset. Aborting.")
        return model
    best_val_loss = inf
    
    loss_history = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Train set: {len(train_dataset)} samples, Validation set: {len(val_dataset)} samples")
    print("-" * 50)
    
    for epoch in range(epochs):
        # --- Training ---
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        num_train_batches = 0
        
        np.random.shuffle(train_dataset)
        
        for i in range(0, len(train_dataset), batch_size):
            batch_data = train_dataset[i:i + batch_size]
            if len(batch_data) < batch_size:
                continue
            
            batch_images = np.stack([x[0] for x in batch_data])
            batch_labels = np.array([x[1] for x in batch_data])
            
            output, cache = forward(batch_images, model)
            loss_value = cross_entropy_loss(output, batch_labels)
            total_train_loss += loss_value
            num_train_batches += 1
            
            predictions = np.argmax(output, axis=1)
            train_correct += np.sum(predictions == batch_labels)
            train_total += len(batch_labels)
            
            gradients = backward(output, batch_labels, cache, model)
            model = update_parameters(model, gradients, learning_rate)
        
        avg_train_loss = total_train_loss / max(num_train_batches, 1)
        train_acc = train_correct / max(train_total, 1)
        
        # --- Validation ---
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        num_val_batches = 0
        
        for i in range(0, len(val_dataset), batch_size):
            batch_data = val_dataset[i:i + batch_size]
            if len(batch_data) == 0: continue
            
            batch_images = np.stack([x[0] for x in batch_data])
            batch_labels = np.array([x[1] for x in batch_data])
            
            output, _ = forward(batch_images, model)
            loss_value = cross_entropy_loss(output, batch_labels)
            total_val_loss += loss_value
            num_val_batches += 1
            
            predictions = np.argmax(output, axis=1)
            val_correct += np.sum(predictions == batch_labels)
            val_total += len(batch_labels)
            
        avg_val_loss = total_val_loss / max(num_val_batches, 1)
        val_acc = val_correct / max(val_total, 1)
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc * 100:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} - Val Acc:   {val_acc * 100:.2f}%")
        
        # Store history
        loss_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        })
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, 'best_model.npz')
            
    # Save to CSV
    df = pd.DataFrame(loss_history)
    df.to_csv(csv_path, index=False)
    print(f"\nLoss history saved to {csv_path}")
    
    print("-" * 50)
    print(f"Training completed! Best Validation Loss: {best_val_loss:.4f}")
    
    return model


# ===================== Testing =====================

def test(model, test_path, image_size=64):
    """
    Evaluate the model on test data.
    """
    test_dataset, _ = load_dataset(test_path, image_size=image_size)
    
    if not test_dataset:
        print("Empty test dataset.")
        return None
    
    total_loss = 0
    correct = 0
    total = 0
    
    batch_size = 32
    for i in range(0, len(test_dataset), batch_size):
        batch_data = test_dataset[i:i + batch_size]
        
        batch_images = np.stack([x[0] for x in batch_data])
        batch_labels = np.array([x[1] for x in batch_data])
        
        output, _ = forward(batch_images, model)
        
        loss_value = cross_entropy_loss(output, batch_labels)
        total_loss += loss_value * len(batch_data)
        
        predictions = np.argmax(output, axis=1)
        correct += np.sum(predictions == batch_labels)
        total += len(batch_labels)
    
    average_loss = total_loss / len(test_dataset)
    accuracy = correct / total
    
    return {
        'average_loss': average_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


# ===================== Main =====================

def main():
    """Main function to run the complete pipeline"""
    print("=" * 60)
    print("6-Layer Image Classification Network with NumPy (5 Hidden Layers)")
    print("=" * 60)
    
    # Paths to the simplified dataset
    base_path = r"e:\learn_midterm\final_classification_dataset"
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'valid')
    test_path = os.path.join(base_path, 'test')
    
    # Hyperparameters
    image_size = 64
    input_size = image_size * image_size
    num_classes = 10
    
    print("\n[1] Initializing model...")
    model = initialize_model(
        input_size=input_size,
        hidden1=4096,
        hidden2=2048,
        hidden3=1024,
        hidden4=512,
        hidden5=256,
        num_classes=num_classes
    )
    print(f"    Input size: {model['config']['input_size']} ({image_size}x{image_size})")
    print(f"    Hidden layers: {model['config']['hidden1']} -> {model['config']['hidden2']} -> {model['config']['hidden3']} -> {model['config']['hidden4']} -> {model['config']['hidden5']}")
    print(f"    Output classes: {model['config']['num_classes']}")
    
    # Train model
    print("\n[2] Training model...")
    model = train(
        model, 
        train_path=train_path, 
        val_path=val_path, 
        image_size=image_size,
        epochs=30, 
        learning_rate=0.005, # Deeper models often need slightly smaller LR
        batch_size=32
    )
    
    # Draw loss
    print("\n[3] Generating loss plot...")
    draw_loss()
    
    # Test model
    print("\n[4] Evaluating model...")
    results = test(model, test_path=test_path, image_size=image_size)
    if results:
        print(f"    Average Loss: {results['average_loss']:.4f}")
        print(f"    Accuracy: {results['accuracy'] * 100:.2f}%")
        print(f"    Correct: {results['correct']}/{results['total']}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return model, results


if __name__ == "__main__":
    model, results = main()
