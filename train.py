import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models
import numpy as np 
from PIL import Image
import json
import argparse

from utilities import parse_train_arguments  # Import the function
from utilities import load_data 
from utilities import build_model
from utilities import train_model
from utilities import test_model
from utilities import save_checkpoint

def main():
      
    args = parse_train_arguments()
    print("Collected CLI arguments")
    
    with open(args.class_mapping, 'r') as f:
        cat_to_name = json.load(f)
    print("Created categories to classes mapping")

    trainloader, validloader, testloader, class_to_idx = load_data(image_dir='flowers', batch_size=64)
    print("Created image loaders")

    model, input_units, output_units = build_model(arch=args.arch, hidden_units=args.hidden_units)
    print("Created the model")

    train_model(model,gpu= args.gpu, epochs = args.epochs , learning_rate = args.learning_rate, trainloader=trainloader, validloader=validloader, print_every=5)
    print("Trained the model")

    test_model(model, testloader, gpu=args.gpu)
    print("Validated the model")
  
    save_checkpoint(model, input_units = input_units, output_units=output_units,hidden_units=args.hidden_units, epochs=args.epochs, checkpoint_path = args.data_dir, arch = args.arch, class_to_idx=class_to_idx)
    print("Saved the model") 



if __name__ == "__main__": main()

