## Usage `train.py`

```python
usage: train.py [-h] [--epochs N] [--learning_rate LR] [--gpu]
                [--hidden_units L1 [L1 ...]] [--arch model]
                [--save_dir /path/to/save]
                data_directory

PyTorch Flower Neural Network

Required named arguments:
  data_directory        Specify the top level data directory i.e. flowers
                                                

Optional named arguments:
  -h, --help            show this help message and exit
                                                
  --epochs N            number of epochs to train (default: 5)
                                                
  --learning_rate LR    learning rate (default: 0.001)
                                                
  --gpu                 Enables CUDA training
                                                
  --hidden_units L1 [L1 ...]
                            Hidden Layers for Network (default: 2 layers of [input_size//2, input_size//4])
                        
  --arch model          Specify the pretrained architecture you wish to load. See: https://pytorch.org/docs/stable/torchvision/models.html
                                                
  --save_dir /path/to/save
                        Specify a directory to save the model.
```

## Usage `predict.py`

```python
usage: predict.py [-h] [--top_k N] [--category_names /path/to/mappings]
                  [--gpu]
                  /path/to/image /path/to/checkpoint

PyTorch Flower Neural Network Inference

Required named arguments:
  /path/to/image        Specify the path to an image file of a flower
                                                
  /path/to/checkpoint   Specify the path to a saved model
                                                

Optional named arguments:
  -h, --help            show this help message and exit
                                                
  --top_k N             The top K predictions to return (default: 1)
                                                
  --category_names /path/to/mappings
                        Specify the path for category to name mappings (default: None)
                                                
  --gpu                 Enables CUDA training
```

