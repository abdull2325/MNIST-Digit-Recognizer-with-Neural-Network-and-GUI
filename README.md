# MNIST-Digit-Recognizer-with-Neural-Network-and-GUI

## Project Overview

This project implements a robust digit recognition system using the MNIST dataset. It features a neural network model with advanced training techniques, cross-validation, hyperparameter tuning, and a graphical user interface (GUI) for real-time digit recognition.

## Key Features

1. **Custom Neural Network Architecture**: Implements a flexible neural network with configurable hidden layers and neurons.
2. **Advanced Training Techniques**: 
   - Utilizes Adam optimizer with learning rate scheduling
   - Implements dropout for regularization
   - Uses LeakyReLU activation function
3. **Hyperparameter Tuning**: Performs grid search to find optimal hyperparameters.
4. **Cross-Validation**: Implements k-fold cross-validation for robust model evaluation.
5. **Data Visualization**: 
   - Plots training and validation loss curves
   - Generates and visualizes confusion matrix
6. **Model Persistence**: Saves the best model during training.
7. **Comprehensive Reporting**: Generates a detailed JSON report of model performance and parameters.
8. **Interactive GUI**: Provides a user interface for drawing digits and getting real-time predictions.

## Unique Achievements

1. **End-to-End Solution**: Combines data loading, preprocessing, model training, evaluation, and deployment in a single script.
2. **Flexibility and Customization**: Allows easy modification of network architecture and hyperparameters.
3. **Robust Evaluation**: Uses cross-validation and grid search for thorough model assessment and optimization.
4. **User-Friendly Interface**: Offers an intuitive GUI for interacting with the trained model.
5. **Comprehensive Documentation**: Provides detailed performance metrics and visualizations.

2. Install required packages:

## Usage Instructions

1. Prepare the MNIST dataset:
- Download the MNIST dataset as a zip file.
- Update the `zip_file_path` variable in the script with the path to your downloaded zip file.

2. Run the main script:
3. The script will perform the following steps:
- Load and preprocess the MNIST data
- Perform cross-validation and hyperparameter tuning
- Train the final model with the best parameters
- Evaluate the model on the test set
- Generate performance reports and visualizations
- Launch the GUI for interactive digit recognition

4. Interact with the GUI:
- Draw a digit on the canvas using your mouse
- Click "Predict" to see the model's prediction
- Use "Clear" to reset the canvas

## Output Files

The script generates several output files:

- `best_model.pth`: The saved best model during training
- `training_progress.png`: Plot of training and validation loss
- `confusion_matrix.png`: Visualization of the confusion matrix
- `model_report.json`: Detailed report of model performance and parameters

## Customization

You can customize the model by modifying the `param_grid` dictionary in the main script. This allows you to experiment with different:

- Number of hidden layers and neurons
- Dropout probabilities
- Learning rates
- Batch sizes
- Number of training epochs

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- The MNIST dataset providers
- PyTorch and scikit-learn communities for their excellent libraries
