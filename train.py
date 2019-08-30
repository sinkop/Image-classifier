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
from collections import OrderedDict

    #DATA DIRECTORIES
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

    #GET ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--dd', type = str, default = 'data_dir', help = '(def)data_dir, your train_data_directory')
parser.add_argument('--cp', type = str, default = 'checkpoints', help = '(def)checkpoints, your_checkpoints_dir (+/arch_name/)')
parser.add_argument('--tr', type = str, default = 't', help = '(def)t:train, r:resume training')
parser.add_argument('--ar', type = str, default = 'densenet161', help = '(def)densenet161, vgg19, alexnet')
parser.add_argument('--hl', nargs='*', type = int, default = None, help = "any hidden layers  --NO ,[]()-- if resuming leave (def)")
parser.add_argument('--dev', type = str, default = 'gpu', help = '(def)GPU if available, else train on CPU')
parser.add_argument('--ep', type = int, default = 30, help = '(def)30, your epochs')
parser.add_argument('--lr', type = int, default = 0.001, help = '(def)0.001, your learning rate')
in_args = parser.parse_args()

data_dir = in_args.dd
checkpoints = in_args.cp
tr = in_args.tr.lower()
arch = in_args.ar
hidden_layers = in_args.hl
epochs = in_args.ep
lr = in_args.lr
epochs_total = 0
output_size = 102
drop_p=0.5
e = 0

    #MESSAGES
if in_args.dev.lower() == 'gpu':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print('turn on GPU!')
        exit()
elif in_args.dev.lower() == 'cpu':
    device = 'cpu'
else:
    print('device MISSPELLED')
    exit()

    #DEFINE CLASSIFIER   
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p):
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
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

    #IF RESUME TRAINING, LOAD BEST CHECKPOINT AND ADDITIONAL INFO FROM FINAL CHECKPOINT
if tr == 'r':
    checkpoint_f = torch.load(checkpoints+'/'+arch+'/'+'checkpoint_f.pth')
    checkpoint_b = torch.load(checkpoints+'/'+arch+'/'+'checkpoint_b.pth')
    arch = checkpoint_f['arch']
    hidden_layers = checkpoint_f['hidden_layers']                             
                                
    #TRANSFORMS
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 

    #LOAD THE DATASETS WITH IMAGEFOLDER
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    #DATALOADERS
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
class_to_idx = train_data.class_to_idx

    #BUILD AND TRAIN MODEL
if arch == 'densenet161':
    model = models.densenet161(pretrained=True)
    input_size = 2208
elif arch == 'vgg19':
    model = models.vgg19(pretrained=True)
    input_size = 25088
elif arch == 'alexnet':
    model = models.alexnet(pretrained=True)
    input_size = 9216
else:
    print('arch MISSPELLED')
    exit()
    
    #REPLACE CLASSIFIER WITH OWN CLASSIFIER
classifier = Network(input_size, output_size, hidden_layers, drop_p)
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

    #IF RESUME TRAINING, LOAD BEST CLASSIFIER DATA AND REBUILD IN NETWORK, ALSO OPTIMIZER DATA
if tr == 'r':
    model.load_state_dict(checkpoint_b['state_dict'])
    model.classifier = checkpoint_b['classifier']
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    optimizer.load_state_dict(checkpoint_b['optimizer'])   

    print('RESUMING training {} {} for {} epochs on {}'.format(arch, hidden_layers, epochs, device))
    print('         from directory:     ', data_dir)
    print('         save checkpoint to: ', checkpoints+'/'+arch+'/')
    print('         learning rate:      ', lr)

    #FREEZE PARAMETERS IN FEATURES, ENABLE PARAMETERS IN CLASSIFIER: JUST TO MAKE SURE REQUIRE=TRUE
for param in model.features.parameters():
    param.requires_grad = False  
for param in model.classifier.parameters():
    param.requires_grad = True
    
if tr == 't':
    print('\n')
    print('TRAINING {} {} for {} epochs on {}'.format(arch, hidden_layers, epochs, device))
    print('         from directory:     ', data_dir)
    print('         save checkpoint to: ', checkpoints+'/'+arch+'/')
    print('         learning rate:      ', lr)
    print('\n')
  
    #DEEP LEARNING
def deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device):
    print_every = print_every
    steps = 0  
    if tr == 't':
        epochs_total = 0
    model.to(device) 
    for e in range(epochs):
        training_loss = 0      
        for ii, data in enumerate(trainloader):
            model.train()
            inputs, labels = data
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)            
            optimizer.zero_grad()      
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
#VALIDATION WHILE TRAINING
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    valid_loss = 0
                    val_accuracy = 0
                    valid_accuracy = 0
                    correct = 0
                    total = 0
                    for images, labels in validloader:   
                        images, labels = images.to(device), labels.to(device)
                        output = model.forward(images)
                        val_loss += criterion(output, labels).item()
                        valid_loss = val_loss/len(validloader)                   
                        if tr == 't' and steps==print_every:
                            best_loss = valid_loss                          #initialize best_loss at step# 40
                        ps = torch.exp(output)
                        _, predicted = torch.max(output.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        equality = (labels.data == ps.max(dim=1)[1])     
                        val_accuracy += equality.type(torch.FloatTensor).mean()
                        valid_accuracy = 100 * val_accuracy/len(validloader)               
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(training_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss),
                      "Validation Accuracy: {:.3f}".format(valid_accuracy))      
                if tr =='r' and steps==print_every:
                    checkpoint_b = torch.load(checkpoints+'/'+arch+'/'+'checkpoint_b.pth')
                    checkpoint_f = torch.load(checkpoints+'/'+arch+'/'+'checkpoint_f.pth')
                    best_loss = checkpoint_b['valid_loss']                  #initialize best_loss for first resume pass
                    epochs_total = checkpoint_f['epochs_total']             #load total epochs from previous training
                if valid_loss < best_loss:
                    best_loss = valid_loss                                  #keep best loss
                    checkpoint_b = {                                        #best checkpoint save
                        'state_dict': model.state_dict(),
                        'classifier': model.classifier,
                        'optimizer': optimizer.state_dict(),
                        'epochs_best': e+1,
                        'training_loss': training_loss/print_every,
                        'valid_loss': valid_loss,
                        'valid_accuracy': valid_accuracy,
                        }
                    torch.save(checkpoint_b, checkpoints+'/'+arch+'/'+'checkpoint_b.pth')                   
                training_loss = 0
                model.train()
    epochs_total += epochs                                                  #total number of epochs from training + resume
    checkpoint_f = {                                                        #final checkpoint save, just some data
                    'arch': arch,
                    'hidden_layers': hidden_layers,
                    'epochs_total': epochs_total,
                    'class_to_idx': class_to_idx,
                    }
    torch.save(checkpoint_f, checkpoints+'/'+arch+'/'+'checkpoint_f.pth')      
deep_learning(model, trainloader, epochs, 40, criterion, optimizer, device)

    #PRINT BEST RESULTS, ADDITIONAL STATS
checkpoint_d = torch.load(checkpoints+'/'+arch+'/'+'checkpoint_b.pth')      #reload best checkpoint saved to print
checkpoint_e = torch.load(checkpoints+'/'+arch+'/'+'checkpoint_f.pth')
tl = checkpoint_d['training_loss']
vl = checkpoint_d['valid_loss']
va = checkpoint_d['valid_accuracy'].item()
et = checkpoint_e['epochs_total']
eb = checkpoint_d['epochs_best']
print('\n\n')
print('     BEST RESULTS:  {}       (not necessarily same as last printed results)\n     ============'.format(arch))
print('Training loss:   ', tl)
print('Validation loss: ', vl)
print('Validation accuracy:', va)
print('     ADDITIONAL STATS:\n     ================')
print('Total epochs run:', et)
print('Best accuracy reached in epoch', et - epochs + eb)
print('\n')
          
    #TESTING THE TRAINED NETWORK WITH 'TEST' FOLDER IMAGES
print('          ....running accuracy measurement....\n')
def check_accuracy_on_test(testloader):    
    correct = 0
    total = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on test images: {}%              (not necessarily same as BEST RESULTS (different test images)'.format(100 * correct / total))
    model.train()
    print('             DONE\n')
check_accuracy_on_test(testloader)







    
    
    
    
    
    
    
    
    
    
    























