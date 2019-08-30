Probably written for Python pre 3.6.7
flowers.tar folder containing the training images is too large to upload
a required empty folder did not upload.  the folder 'checkpoints' contains three subfolders, the structure is as follows:  checkpoints/alexnet; checkpoints/densenet161; checkpoints/vgg19

TRAIN can train first time or resume, if resume pick arch to be resumed and leave hidden_layers at default (will reload approppriate hidden_layers)

hl: enter any number of hidden layers, do not use commas () or []
hidden_layers used:
	densenet161:  	[1104, 552, 276]
	alexnet:	[4608, 2304, 1152]
	vgg19:		[12544, 6272, 3136]

vgg19 crashes due to unknown error


	*** UPDATE: ***

I tested vgg19, a simple (separate file) build/save, then reload/rebuild works.  the same process still doesn't work in train.py.  the unknown error occurs when saving the first best checkpoint (checkpoint_b), the file appears in the folder but doesn't completely save, since it separately works but it doesn't work here, the problem may be with saving the calculation results (and not saving network weights)
