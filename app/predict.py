import torch
import argparse
from argparse import RawTextHelpFormatter
import json
from utils import process_image, load_checkpoint
from train import CustomClassifier
import os

def get_class_names(path):
    """
    This function returns the JSON representation of category labels to class names

    Inputs:
        - path: The path to the JSON file containing the mapping
    
    Outputs:
        - cat_to_name: JSON object representing the mapping
    """

    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    f.close()
    
    return cat_to_name

def predict(image_path, model, device, topk):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Inputs:
        - image_path: path to image that will be passed to the model.
        - model: pretrained model
        - topk: Will display top k probabilities for a given image. Defaults to k=5.
    
    Outputs:
        - top_probs: The probabilities associated with predictions
        - classes: The category labels associated with the predictions
    '''

    #single image - need to insert a batch axis for model compatibility, hence we use unsqueeze on dimension 0.
    image = torch.tensor(process_image(image_path)).type(torch.FloatTensor).unsqueeze_(0).to(device)
    
    model.to(device)
    
    #turn off dropout
    model.eval()
    
    # reverse class to idx dict so that an index returns a class
    class_map = {v:k for k,v in model.class_to_idx.items()}
    
    # no gradient tracking
    with torch.no_grad():
        
        output = model.forward(image)
        
        probability = torch.exp(output)
        
        top_probs, indicies = probability.topk(topk, dim=1)
    
    if topk != 1:

        classes = [class_map[x] for x in indicies.squeeze().tolist()]
        
        classes, top_probs = zip(*sorted(zip(classes,top_probs.squeeze().tolist()), reverse=True, key=lambda x: x[1]))
    else:
        index = indicies.squeeze().tolist()
        classes = class_map[index]
        top_probs = top_probs.squeeze().tolist()
    
    return top_probs, classes

def main():
    parser = argparse.ArgumentParser(description='PyTorch Flower Neural Network Inference', formatter_class=RawTextHelpFormatter, add_help=False)
    requiredNamed = parser.add_argument_group('Required named arguments')
    optionalNamed = parser.add_argument_group('Optional named arguments')
    requiredNamed.add_argument('input', type=str, metavar='/path/to/image',
                        help='''Specify the path to an image file of a flower
                        ''')
    requiredNamed.add_argument('checkpoint', type=str, metavar='/path/to/checkpoint',
                        help='''Specify the path to a saved model
                        ''')
    optionalNamed.add_argument('-h','--help', action='help',
                        help='''show this help message and exit
                        ''')
    optionalNamed.add_argument('--top_k', type=int, default=1 ,metavar='N',
                        help='''The top K predictions to return (default: 1)
                        ''')
    optionalNamed.add_argument('--category_names', type=str, default=None, metavar='/path/to/mappings',
                        help='''Specify the path for category to name mappings (default: None)
                        ''')
    optionalNamed.add_argument('--gpu', action='store_true', default=False,
                        help='''Enables CUDA training
                        ''')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise ValueError(f"No image file found at: {args.input}")
    
    if not os.path.exists(args.checkpoint):
        raise ValueError(f'No checkpoint found at: {args.checkpoint}')
    
    top_k = args.top_k
    
    if args.gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    loaded_model = load_checkpoint(args.checkpoint, CustomClassifier, device)
    
    probs, classes = predict(args.input, loaded_model, device, top_k)

    if args.category_names:

        cat_to_name = get_class_names(args.category_names)
    
        if top_k != 1:

            class_names = [cat_to_name[x] for x in list(classes)]
        else:
            class_names = cat_to_name[classes]

        print("The probabilties are: ", probs)
        print()
        print('The associated flower names are: ', class_names)
    else:
        print("The probabilties are: ", probs)
        print()
        print('The associated class labels: ', classes)


if __name__ == '__main__':
    main()
