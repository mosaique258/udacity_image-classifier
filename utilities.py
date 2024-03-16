import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models
import numpy as np 
from PIL import Image

import json

def parse_train_arguments():
  """Parses command line arguments for training script.

  Returns:
      argparse.Namespace: An object containing parsed arguments.
  """

  # Create the argument parser
  parser = argparse.ArgumentParser(description="Train an image classification model")

  # Required argument: data directory
  parser.add_argument("--data_dir", type=str, default="cli_checkpoint.pth",
                      help="Path to the directory containing the image data")

  # Model architecture (restricted to vgg or densenet)
  parser.add_argument("--arch", type=str, choices=["vgg", "densenet"], default="densenet",
                      help="The architecture of the model to use (either 'vgg' or 'densenet')")

  # Learning rate
  parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Learning rate for the optimizer (default: 0.001)")

  # Number of hidden units
  parser.add_argument("--hidden_units", type=int, default=256,
                      help="Number of hidden units in the final layer (default: 256)")

  # Training epochs
  parser.add_argument("--epochs", type=int, default=2,
                      help="Number of epochs to train the model (default: 5)")

  # Use GPU
  parser.add_argument("--gpu", action="store_true", default=False,
                      help="Use GPU for training (default: False)")

  # Class mapping JSON file
  parser.add_argument("--class_mapping", type=str, default='cat_to_name.json',
                      help="Path to a JSON file mapping class values to categories")

  
 

  # Parse arguments
  args = parser.parse_args()

  return args




def load_data(image_dir, batch_size=64):
    """Loads the training, test and validation data"""

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(image_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(image_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(image_dir + '/valid', transform=test_transforms)


    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size) 

    # Get the mapping of class names to integer labels from the training data
    class_to_idx = train_data.class_to_idx
 

    return trainloader, validloader, testloader, class_to_idx

def build_model(arch, hidden_units):
    """Builds the model architecture"""

    output_units = 102

    if arch == "vgg":
        model = models.vgg13(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        
        input_units= 25088

        
        model.classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, output_units),
                                 nn.LogSoftmax(dim=1))
        

    else:
        model = models.densenet121(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        input_units= 1024
        

        model.classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_units, output_units),
                                    nn.LogSoftmax(dim=1))
                

    print(f"{arch} model built with {hidden_units} hidden units.")
    
    return model, input_units, output_units 

def train_model(model,gpu, epochs , learning_rate, trainloader, validloader, print_every=5):
    
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Set device based on GPU availability and user preference
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    # Move model and tensors to device
    model = model.to(device)

    # Train the network

    #epochs = 2
    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                batch_step = (steps // print_every)  # Integer division for batch step number
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Batch Step: {batch_step}.. "  #/ {batch_steps_print} Added line for batch step
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {test_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
            


def test_model(model,testloader,gpu):
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables for accuracy calculation
    correct = 0
    total = 0

    # Set device based on GPU availability and user preference
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over the test dataset
        for images, labels in testloader:
            # Move images and labels to the device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get the predicted labels
            _, predicted = torch.max(outputs.data, 1)
            
            # Update total count
            total += labels.size(0)
            
            # Update correct count
            correct += (predicted == labels).sum().item()

    # Calculate the accuracy
    accuracy = 100 * correct / total

    # Print the accuracy
    print('Accuracy on the test set: {:.2f}%'.format(accuracy))


def save_checkpoint(model, input_units, output_units, hidden_units, epochs, checkpoint_path, arch, class_to_idx):
    model.class_to_idx = class_to_idx
    checkpoint = {'state_dict': model.state_dict(),
                'input_units': input_units,
                'output_units': output_units,
                'hidden_layers': hidden_units,
                'class_to_idx': model.class_to_idx,
                'epochs': epochs,
                'arch': arch,  

    }
              

    torch.save(checkpoint, checkpoint_path)


def parse_predict_arguments():
  """Parses command line arguments for prediction script.

  Returns:
      argparse.Namespace: An object containing parsed arguments.
  """

  # Create the argument parser
  parser = argparse.ArgumentParser(description="Predict image with a trained neural network")



  # Use GPU
  parser.add_argument("--gpu", action="store_true", default=False,
                      help="Use GPU for training (default: False)")

  # Class mapping JSON file
  parser.add_argument("--class_mapping", type=str, default=' ',
                      help="Path to a JSON file mapping class values to categories")

   # Save directory for checkpoints (optional)
  parser.add_argument("--save_dir", type=str, default="./checkpoints", nargs='?',
                       help="Directory to save training checkpoints (default: ./checkpoints)")
  
  parser.add_argument("--checkpoint_loc", type=str, default="./checkpoints",
                      help="Location from where to read checkpoint file" )
  
  parser.add_argument("--inf_img_path", type=str, default="image.jpg", 
                      help="Path to the image to be classified")
  
  parser.add_argument  ("--topk", type=int, default=5,
                        help="Number of top predictions to return (default: 5)")
  
 

  # Parse arguments
  args = parser.parse_args()

  return args

def load_checkpoint(checkpoint_location, gpu, arch, hidden_units):
    checkpoint = torch.load(checkpoint_location)

    model, input_units, output_units = build_model(checkpoint['arch'], checkpoint['hidden_units'])
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Define the transformations
    preprocess = transforms.Compose([
    transforms.Resize(256),  # Resize to closest side to 256 pixels (keeping aspect ratio)
    transforms.CenterCrop(224),  # Crop out a 224x224 portion from the center
    transforms.ToTensor(),  # Encode color channels to 0-1 floats and normalize the image
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ])

    # Load and preprocess the image
    image = Image.open(image)  # Replace 'image.jpg' with the path to your image
    preprocessed_image = preprocess(image)
    
    # Transpose the color channel from the third to the first dimension
    #preprocessed_image = preprocessed_image.transpose(0, 2)

    # Verify the shape of the preprocessed image
    print(preprocessed_image.shape)

    return preprocessed_image

def predict(image, model, k, gpu):
    
    """
    
    Predicts the top k most probable classes for an image.

    Args:
        image_path (str): Path to the image file.
        model (torch.nn.Module): PyTorch model for classification.
        k (int, optional): Number of top predictions to return. Defaults to 5.

    Returns:
        tuple: Tuple containing two elements:
            - probs (torch.Tensor): Tensor of top k probabilities (between 0 and 1).
            - classes (list): List of top k predicted class labels.
    """
    
    # Preprocess the image
    preprocessed_image = process_image(image)
    
    # Add a batch dimension for the model
    preprocessed_image = preprocessed_image.unsqueeze(0)
    
    # Set device based on GPU availability and user preference
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    # Move model and tensors to device
    model = model.to(device)

    # Set model to evaluation mode (optional for some models)
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(preprocessed_image)
    
    # Apply softmax to get probabilities between 0 and 1
    probs = F.softmax(outputs, dim=1)
    
    # Get top k probabilities and indices
    topk_probs, indices = probs.topk(k, dim=1)
    
    # Squeeze the first dimension (batch size)
    topk_probs = topk_probs.squeeze(0)
    
    #Invert class
    inverted_class_to_idx = {value: key for key, value in model.class_to_idx.items()}
    classes = [inverted_class_to_idx[i.item()] for i in indices.squeeze()]  # Squeeze to remove extra dimension
    

    return topk_probs, classes
