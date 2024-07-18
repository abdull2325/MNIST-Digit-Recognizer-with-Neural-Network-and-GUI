import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
import zipfile
import struct
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import itertools
import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL import Image, ImageTk
import numpy as np
import io
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import torch
import io
import tkinter as tk
from PIL import Image, ImageGrab
import numpy as np
import torch
# Load the MNIST dataset from the provided zip file
def load_mnist_from_zip(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        files = z.namelist()
        train_images_path = [f for f in files if 'train-images' in f][0]
        train_labels_path = [f for f in files if 'train-labels' in f][0]
        test_images_path = [f for f in files if 't10k-images' in f][0]
        test_labels_path = [f for f in files if 't10k-labels' in f][0]

        train_images = extract_images(z.open(train_images_path))
        train_labels = extract_labels(z.open(train_labels_path))
        test_images = extract_images(z.open(test_images_path))
        test_labels = extract_labels(z.open(test_labels_path))

    return (train_images, train_labels), (test_images, test_labels)


def extract_images(file):
    with file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        images = np.frombuffer(file.read(), dtype=np.uint8).reshape(num, 784)
        return images / 255.0


def extract_labels(file):
    with file:
        magic, num = struct.unpack(">II", file.read(8))
        labels = np.frombuffer(file.read(), dtype=np.uint8)
        return labels


class NeuralNetwork(nn.Module):
    def __init__(self, layers, dropout_prob=0.5):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.LeakyReLU(0.01))
                self.layers.append(nn.Dropout(dropout_prob))
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train(model, train_loader, val_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_accuracy = 0
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        training_losses.append(avg_loss)

        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.4f}, Val Loss: {avg_val_loss:.4f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()

    plot_training_progress(training_losses, validation_losses)
    return best_accuracy, training_losses, validation_losses


def plot_training_progress(training_losses, validation_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.savefig('training_progress.png')
    plt.close()


def prepare_data(images, labels):
    images_tensor = torch.FloatTensor(images).view(-1, 784) / 255.0
    labels_tensor = torch.LongTensor(labels)
    return TensorDataset(images_tensor, labels_tensor)


def grid_search(dataset, param_grid, device):
    best_accuracy = 0
    best_params = None
    results = []

    for params in itertools.product(*param_grid.values()):
        current_params = dict(zip(param_grid.keys(), params))
        print(f"Testing parameters: {current_params}")

        model = NeuralNetwork([784] + [current_params['hidden_neurons']] * current_params['num_hidden_layers'] + [10],
                              dropout_prob=current_params['dropout_prob']).to(device)

        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

        train_loader = DataLoader(train_dataset, batch_size=current_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=current_params['batch_size'], shuffle=False)

        accuracy, _, _ = train(model, train_loader, val_loader,
                               epochs=current_params['epochs'],
                               learning_rate=current_params['learning_rate'],
                               device=device)

        results.append((current_params, accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = current_params

    return best_params, results


def cross_validate(dataset, k_folds, param_grid, device):
    kf = KFold(n_splits=k_folds, shuffle=True)
    cv_results = []

    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{k_folds}")
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        best_params, _ = grid_search(train_dataset, param_grid, device)

        model = NeuralNetwork([784] + [best_params['hidden_neurons']] * best_params['num_hidden_layers'] + [10],
                              dropout_prob=best_params['dropout_prob']).to(device)

        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)

        accuracy, _, _ = train(model, train_loader, val_loader,
                               epochs=best_params['epochs'],
                               learning_rate=best_params['learning_rate'],
                               device=device)

        cv_results.append((best_params, accuracy))

    return cv_results


def generate_report(cv_results, best_model_accuracy, test_accuracy, cm):
    report = {
        "cross_validation_results": cv_results,
        "best_model_accuracy": best_model_accuracy,
        "test_accuracy": test_accuracy,
        "confusion_matrix": cm.tolist()
    }

    with open('model_report.json', 'w') as f:
        json.dump(report, f, indent=4)

    print("Report generated and saved as 'model_report.json'")


class DigitRecognizerGUI:
    def __init__(self, master, model, device):
        self.master = master
        self.model = model
        self.device = device
        master.title("Digit Recognizer")

        self.canvas = tk.Canvas(master, width=280, height=280, bg="black")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_button = tk.Button(master, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", width=0)

    def predict_digit(self):
        # Get the canvas coordinates
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        # Capture the canvas area
        img = ImageGrab.grab(bbox=(x, y, x1, y1))

        # Resize and convert the image
        img = img.resize((28, 28)).convert('L')
        img_array = np.array(img).reshape(1, 784) / 255.0
        img_tensor = torch.FloatTensor(img_array).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            prediction = output.argmax(dim=1).item()

        self.result_label.config(text=f"Predicted digit: {prediction}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="")


# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    zip_file_path = r'C:\Users\hp\PycharmProjects\pythonProject9\mini\archive (4).zip'
    (train_images, train_labels), (test_images, test_labels) = load_mnist_from_zip(zip_file_path)

    full_dataset = prepare_data(np.concatenate((train_images, test_images)),
                                np.concatenate((train_labels, test_labels)))

    # Define parameter grid for grid search
    param_grid = {
        'hidden_neurons': [64, 128],
        'num_hidden_layers': [2, 4],
        'dropout_prob': [0.3, 0.5],
        'epochs': [30, 50],
        'batch_size': [64, 128],
        'learning_rate': [0.001, 0.0001]
    }

    # Perform cross-validation
    cv_results = cross_validate(full_dataset, k_folds=5, param_grid=param_grid, device=device)

    # Get the best parameters
    best_params = max(cv_results, key=lambda x: x[1])[0]

    # Train the final model with the best parameters
    train_size = int(0.8 * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,
                                                               [train_size, len(full_dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)

    final_model = NeuralNetwork([784] + [best_params['hidden_neurons']] * best_params['num_hidden_layers'] + [10],
                                dropout_prob=best_params['dropout_prob']).to(device)

    best_accuracy, _, _ = train(final_model, train_loader, val_loader,
                                epochs=best_params['epochs'],
                                learning_rate=best_params['learning_rate'],
                                device=device)

    # Evaluate on test set
    final_model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = final_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Generate and save report
    generate_report(cv_results, best_accuracy, test_accuracy, cm)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Launch GUI
    root = tk.Tk()
    gui = DigitRecognizerGUI(root, final_model, device)
    root.mainloop()
