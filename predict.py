import argparse
from utilities import load_checkpoint
from utilities import predict
import json
from utilities import parse_predict_arguments  # Import the function

def main():

    args = parse_predict_arguments()
    print("Collected CLI arguments")

    with open(args.class_mapping, 'r') as f:
        cat_to_name = json.load(f)
    print("Created categories to classes mapping")

    prediction_model = load_checkpoint(checkpoint_location=args.checkpoint_loc)
    print('Loaded the model from checkpoint')

    probs, classes = predict(model= prediction_model, image=args.inf_img_path, k=args.top_k, gpu=args.gpu)
    print(classes)
    print(probs)

if __name__ == "__main__": main()