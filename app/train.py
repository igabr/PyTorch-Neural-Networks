import argparse
from argparse import RawTextHelpFormatter
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import sys

from utils import save_model, load_checkpoint, validation


class CustomClassifier(nn.Module):
    """
    This is a class representing my Custom Classifier.
    
    This custom classifier will replace the classifiers that are present in pre-trained models.
    
    This custom classifier has a dropout layer between every hidden layer.
    
    Inputs:
        input_size: The dimension of the matrix/feature space that will be fed into the **classifier**.
        output_size: The dimension corresponding the the number of classes - i.e. the output of the classifier.
        hidden_layers: A list representing the dimensions of the intermediate hidden layers.
        drop_p: Probability of an element to be zeroed when passing through a dropout layer.
        
    """
    def __init__(self, input_size, output_size, hidden_layers, drop_p):
        super().__init__()
        
        assert type(input_size) == int, "Enter input size as an integer"
        assert type(output_size) == int, "Enter output size as an integer"
        assert type(hidden_layers) == list, "Enter hidden layers as a list e.g. [512,256,...]"
        assert all(isinstance(x, int) for x in hidden_layers), "Enter hidden layers and integers"
        assert type(drop_p) == list, "Enter dropout ratios per layer as a list"
        assert len(drop_p) == len(hidden_layers), "Please enter a dropout probability per hidden layer"
        assert all(isinstance(x, float) for x in drop_p), "Enter dropout ratios as floats"
        assert len([x for x in drop_p if 0 <= x <= 1]) == len(drop_p), "All drop out ratios must be in [0,1]"
            
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        self.layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        
        self.hidden_layers.extend([nn.Linear(h1,h2) for h1,h2 in self.layer_sizes])
        
        # we put this in a seperate module list as the forward pass uses a for-loop to process a sample x.
        self.dropout = nn.ModuleList([nn.Dropout(p=x) for x in drop_p])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
    
    def forward(self, x):
        for index, linear in enumerate(self.hidden_layers):
            x = F.relu(linear(x)) # pass through hidden layer
            x = self.dropout[index](x) # pass through dropout
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1) # straight to NLLLoss


def train(model, output_size, optimizer_name, trainloader, validationloader, epochs, learning_rate, print_every, device, hidden_layers=None):
    """
    All models used with this training function will have a dropout layer between every hidden layer with p=0.1
    
    Inputs:
        - model: The model object that requires training.
        - output_size: The output size of the network - corresponds to the number of classes
        - optimizer_name: The string name of the optimizer to use i.e. Adam
        - trainloader: DataLoader object representing the training data
        - validationloader: DataLoader object representing the validation data
        - epochs: Integer representing the number of epochs the model should run through.
        - learning_rate: Float representing the learning rate i.e. 0.001
        - print_every: Integer representing the frequency of printed statistics i.e. Every 50 iterations
        - hidden_layers
            - Defaults to None, this creates a default CustomClassifier that will be added to the pretraoned model.
            - Otherwise, enter a list representing the hidden layer dimensions to be used for the CustomClassifier.
    Outputs:
        - model: Trained model
        
    """
    
    model.to(device) # run on GPU
    
    # freeze entire pre-loaded Network.
    for param in model.parameters():
        param.requires_grad = False
    
    ### Building out Custom Classifier Depending on pretrained network
    input_size = list(model.classifier.named_children())[0][-1].in_features # input size to custom network
    
    if hidden_layers:
        
        new_classifier = CustomClassifier(input_size, output_size, hidden_layers, [0.1]*len(hidden_layers))
    else:
        # default hidden layers
        default_layers = [input_size//2,input_size//4]
        # dropout probs set to 0.1 per layer
        new_classifier = CustomClassifier(input_size, output_size, default_layers, [0.1]*len(default_layers))
    
    
    new_classifier.to(device)
    
    # update pretrained model to use new classifier - transfer learning step.
    model.classifier = new_classifier # parameters for this classifier are NOT frozen.

    criterion = nn.NLLLoss()
    
    #optimizer only concerned with classifier.
    optimizer = eval(f"optim.{optimizer_name}(model.classifier.parameters(), lr={learning_rate})")
    
    steps = 0
    running_loss = 0
    
    for e in range(epochs):
        model.train() # train mode.
        
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device) # GPU copy
            
            # No need for image resize as the image is going through a pre-trained network.
            
            optimizer.zero_grad() # zero out the gradient calculations for every batch
            
            output = model.forward(images)
            
            loss = criterion(output, labels)
            
            loss.backward() # backprop over classifier parameters
            
            optimizer.step() # take step
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval() # removes dropout
                
                # no gradient tracking here.
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validationloader, criterion, device)
                    
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss),
                          "Validation Accuracy: {:.3f}".format(accuracy))
                
                running_loss = 0
                
                model.train() # enter training mode again.
    
    return model


def main():
    
    parser = argparse.ArgumentParser(description='PyTorch Flower Neural Network', formatter_class=RawTextHelpFormatter, add_help=False)
    requiredNamed = parser.add_argument_group('Required named arguments')
    optionalNamed = parser.add_argument_group('Optional named arguments')
    requiredNamed.add_argument('dir', type=str, metavar='data_directory',
                        help='''Specify the top level data directory i.e. flowers
                        ''')
    optionalNamed.add_argument('-h','--help', action='help',
                        help='''show this help message and exit
                        ''')
    optionalNamed.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='''number of epochs to train (default: 5)
                        ''')
    optionalNamed.add_argument('--learning_rate',type=float, default=0.001, metavar='LR',
                        help='''learning rate (default: 0.001)
                        ''')
    optionalNamed.add_argument('--gpu', action='store_true', default=False,
                        help='''Enables CUDA training
                        ''')
    optionalNamed.add_argument('--hidden_units', nargs='+', type=int, default=None, metavar='L1',
                        help='''    Hidden Layers for Network (default: 2 layers of [input_size//2, input_size//4])

''')
    optionalNamed.add_argument('--arch', type=str, default='vgg11', metavar='model',
                        help='''Specify the pretrained architecture you wish to load. See: https://pytorch.org/docs/stable/torchvision/models.html
                        ''')
    optionalNamed.add_argument('--save_dir', type=str, metavar='/path/to/save',
                        help='Specify a directory to save the model.')
    
    args = parser.parse_args()

    #### Global Variables
    OUTPUT_SIZE = 102
    OPTIMIZER = 'Adam'
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    print_every = 50
    #### End Global Variables

    if args.gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'

    try:
        model = eval(f"models.{args.arch}()")
    except AttributeError as e:
        raise AttributeError(f"{args.arch} is not a valid model") from e
    
    if hasattr(model, 'classifier'):
        model = eval(f"models.{args.arch}(pretrained=True)")
    else:
        raise ValueError("Please enter a model that has a classifier attribute.")
    

    ### DIRECTORIES
    data_dir = args.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    ### END DIRECTORIES
    
    ### DataLoaders
    data_transforms = {'training': transforms.Compose([transforms.RandomRotation(45), transforms.RandomResizedCrop(224), transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(means, stds)]),
                    'validation': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(means, stds)])}

    image_datasets = {'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
                    'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])}

    dataloaders = {"training": DataLoader(image_datasets['training'], batch_size=32, shuffle=True),
                "validation": DataLoader(image_datasets['validation'], batch_size=32, shuffle=True)}
    ### End DataLoaders

    trained_model = train(model, OUTPUT_SIZE, OPTIMIZER, dataloaders['training'], dataloaders['validation'], args.epochs, args.learning_rate, print_every, device, hidden_layers=args.hidden_units)

    if args.save_dir:
        save_model(args.arch, trained_model, image_datasets['training'], args.save_dir)

if __name__ == '__main__':
    main()