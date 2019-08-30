    #IMPORTS
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import torchvision
import torchvision.transforms as transforms
import argparse
import json
from collections import OrderedDict
from PIL import Image
from decimal import Decimal

    #GET ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--pd', type = str, default = 'pred_im_folder', help = '(def)pred_im_folder, your image folder')
parser.add_argument('--image', type = str, default = 'image_06743.jpg', help = '(def)image_06743.jpg, your image.jpg')
parser.add_argument('--cp', type = str, default = 'checkpoints', help = '(def)checkpoints, your_checkpoints_dir (+/arch_name/)')
parser.add_argument('--ar', type = str, default = 'densenet161', help = '(def)densenet161, vgg19, alexnet')
parser.add_argument('--dev', type = str, default = 'gpu', help = '(def)GPU IS NEEDED!')
parser.add_argument('--topk', type = int, default = 5, help = 'top K classes and probabilities')
parser.add_argument('--jf', type = str, default = 'cat_to_name.json', help = '(def)cat_to_name.json, your JSON.json file')
in_args = parser.parse_args()

pred_im_folder = in_args.pd
checkpoints = in_args.cp
image = in_args.image
arch = in_args.ar
topk = in_args.topk
jf =in_args.jf
output_size = 102
drop_p=0.5

    #MESSAGES
with open(jf, 'r') as f:
    cat_to_name = json.load(f)
if in_args.dev.lower() == 'gpu':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print('turn on GPU!')
        exit()      
elif in_args.dev.lower() == 'cpu':
    print('switch to GPU!')
    exit()
else:
    print('DEVICE MISSPELLED')  
    exit()
    
    #DEFINE CLASSIFIER
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.

        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)        
        x = self.output(x)        
        return F.log_softmax(x, dim=1)
    
        #LOAD BEST SAVED CHECKPOINT DEPENDING ON CHOSEN ARCH
checkpoint_b = torch.load(checkpoints+'/'+arch+'/'+'checkpoint_b.pth')
checkpoint_f = torch.load(checkpoints+'/'+arch+'/'+'checkpoint_f.pth')
hidden_layers = checkpoint_f['hidden_layers']
arch = checkpoint_f['arch']

if arch == 'densenet161':
    model = models.densenet161(pretrained=True)
    input_size = 2208
elif arch == 'vgg19':
    model = models.vgg19(pretrained=True)
    input_size = 25088
else: 
    arch == 'alexnet'
    model = models.alexnet(pretrained=True)
    input_size = 9216    

    #REBUILD CLASSIFIER
classifier = Network(input_size, output_size, hidden_layers, drop_p)
model.classifier = classifier
model.load_state_dict(checkpoint_b['state_dict'])
model.classifier = checkpoint_b['classifier']
model.class_to_idx = checkpoint_f['class_to_idx']

    #PROCESS IMAGE
def process_image(image):
    pil_im = Image.open(image)
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    p_pil_im = test_transforms(pil_im)
    return p_pil_im
print('\n\n\n')
print('PREDICTING: ', pred_im_folder+'/'+image, ' top{} probabilities using {} and {} on {} ....'.format(topk, arch, jf, device))
print('\n')
      
    #PREDICT
def predict(image_path, model, topk):
    model.eval()
    model.to(device)
    p_pil_im = process_image(image_path)
    p_pil_im.unsqueeze_(0)
    p_pil_im = p_pil_im.to(device)
    output = model(p_pil_im)
    probs, class_index = output.topk(topk)
    probs = probs.cpu()
    class_index = class_index.cpu()
    probs = probs.exp().data.numpy()[0]
    class_index = class_index.data.numpy()[0]
    class_keys = {x: y for y, x in model.class_to_idx.items()}
    classes = [class_keys[i] for i in class_index]   
    names = [cat_to_name.get(i) for i in classes[::]]
    return probs, class_index, classes, names
probs, class_index, classes, names = predict(pred_im_folder+'/'+image, model, topk)
      
prob = round(probs[0], 2)
print('     RESULTS    {}\n     ======='.format(arch))
print('Class indexes:           ', class_index)
print('Mapped classes:          ', classes)
print('Top {} probabilities:     '.format(topk), probs)
print('Top {} flower names:      '.format(topk), names)
print('\n')
print('PREDICTION: This flower is', prob,'% likely a {}'.format(names[0]))
print('\n                   DONE\n')

      
    

