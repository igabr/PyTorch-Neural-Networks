import torchvision
from torchvision import models
import torch
from PIL import Image
import numpy as np

def save_model(model_name, model, training_img_folder, filename):
    """
    This specific function is used to save a CustomClassifier object that is used in transfer learning.
    
    Inputs:
        - model_name: Name of the Transfer Learning Architecture (i.e. vgg11)
        - model: Transfer Learning Model
        - training_img_folder: The Training data ImageFolder object
        - filename: Name of saved model file to use. Format: filename_{model_name}.pth
    
    Outputs:
        - None
    """
    
    assert type(training_img_folder) == torchvision.datasets.folder.ImageFolder, "Load the ImageFolder Object associated with training data"
    assert hasattr(model, 'classifier') == True, "Pass a model that has a classifier attribute."
        
    clf = model.classifier

    checkpoint = {}

    checkpoint['input_size'] = clf.hidden_layers[0].in_features
    checkpoint['hidden_layers'] = [each.out_features for each in clf.hidden_layers]
    checkpoint['dropout'] = [each.p for each in clf.dropout]
    checkpoint['output_size'] = clf.output.out_features
    checkpoint['state_dict'] = clf.state_dict()
    checkpoint['model_architecture'] = model_name
    checkpoint['class_to_idx'] = training_img_folder.class_to_idx

    if model_name not in filename:
        filename += f'_{model_name}'

    if ".pth" not in filename:
        filename += '.pth'

    torch.save(checkpoint, filename)

    print(f"Model with has been saved to a file called {filename}")

def load_checkpoint(filepath, clf_class, device):
    """
    This specific function is used to load a checkpoint that 
    represents a clf_class object that is used in transfer learning.
    
    All params have gradients frozen.
    
    Inputs:
        - Filepath of checkpoint
        - clf_class: The object with which the classifier in the pretrained network was replaced with (i.e. CustomClassifier)
        - device: The device the model should be loaded on to. This is a string
    
    Output:
        - model: Transfer Learning model loaded with a clf_class.
    """

    cuda = torch.cuda.is_available()

    if cuda:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, location: device) #loads a GPU trained model onto a CPU

    
    new_classifier = clf_class(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'],
                             checkpoint['dropout'])
    
    #loading classifier state dict
    new_classifier.load_state_dict(checkpoint['state_dict'])
    
    # Transfer Architecture type used in creating the classifier
    model_arch = checkpoint['model_architecture']
    
    # Reload that architecture
    model = eval(f"models.{model_arch}(pretrained=True)")
    
    # Replace default classifier with saved classifier
    model.classifier = new_classifier
    
    #model to device
    model.to(device)

    # storing the training data class to idx attribute
    model.class_to_idx = checkpoint['class_to_idx']
    
    # freeze entire Network - no more training to occur.
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def validation(model, validationloader, criterion, device):
    """
    During training, we will look at the performance of our Network with regards to the Validation Set.
    
    Inputs:
        - model: The model that is currently being trained.
        - validationloader: A dataloader object representing the validation data.
        - criterion: The criterion used to train the model.
    """
    
    validation_loss = 0
    
    validation_accuracy = 0
    
    model.to(device)
    
    for images, labels in validationloader:
        
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        
        validation_loss += criterion(output, labels).item()
        
        probability = torch.exp(output)
        
        equality = (labels.data == probability.max(dim=1)[1])
        
        validation_accuracy += equality.type(torch.FloatTensor).mean()
    
    validation_loss = validation_loss/len(validationloader)
    validation_accuracy = validation_accuracy/len(validationloader)
    
    return validation_loss, validation_accuracy

def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array

    Inputs:
        - image_path: path to image to be processed
    
    Returns:
        - image: The processed Image Object.
    '''
    image = Image.open(image_path)

    original_width, original_height = image.size

    shortest_side = min(original_width, original_height)

    # Maintain Aspect Ratio
    new_width, new_height = int((original_width/shortest_side)*256), int((original_height/shortest_side)*256)
    image = image.resize((new_width, new_height))
    
    width, height = image.size
    
    #Center Cut
    center_width = 224
    center_height = 224
    
    left_side_move = (width - center_width)/2
    top_side_move = (height - center_height)/2
    right_side_move = (width + center_width)/2
    bottom_side_move = (height + center_height)/2
    
    # Crop according to these pixel locations that make a box.
    image = image.crop((left_side_move, top_side_move, right_side_move, bottom_side_move))

    # 0-1 range color channels.
    image = np.array(image)
    image = image/255 # broadcasting

    # Match Model Normalization Criteria
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean) / std # broadcasting

    # Reordering
    image = image.transpose((2, 0, 1))

    return image