import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import torch
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE, Isomap
import glob
from astropy.visualization import simple_norm
import os
#from torchvision.io import read_image
from tqdm import tqdm
import pandas as pd
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchsummary import summary
from SetRandomSeed import set_random_seeds
from skimage.transform import resize
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier


set_random_seeds(626)
CNNName = 'Adam_Cyclic'
#class_for_gradcam = 'nonmerger'
class_for_gradcam = 'merger'

# Load the saved model state
checkpoint = torch.load('ResNet_' + CNNName +'.pth', map_location=torch.device('cpu'))


learning_rate = 1e-5
lr_decay = 0.5

model = FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-resnet18', learning_rate=learning_rate,  # use a low learning rate
    lr_decay=lr_decay,  # reduce the learning rate from lr to lr^0.5 for each block deeper in the network
    # arguments specific to FinetuneableZoobotClassifier
    num_classes=2
)
# Load the model weights from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to('cpu')  # Ensure the model is on the correct device (GPU/CPU)
model.eval()  # Switch to evaluation mode

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from BinaryMergerDataset import BinaryMergerDataset, get_transforms
path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1widebinmocks/'
BATCH_SIZE = 64

test_mergers_dataset_orig = BinaryMergerDataset(path, 'test', mergers = True, transform = get_transforms(aug=False), codetest=False)
test_nonmergers_dataset_orig = BinaryMergerDataset(path, 'test', mergers = False, transform = get_transforms(aug=False), codetest=False)

test_dataset_full = torch.utils.data.ConcatDataset([test_mergers_dataset_orig, test_nonmergers_dataset_orig])
test_dataloader = DataLoader(test_dataset_full, shuffle = True, num_workers = 0, batch_size=BATCH_SIZE)#num workers used to be 4

all_labels = []
all_preds = []
all_names = []
all_probabilities = []

with torch.no_grad():  # No need to track gradients during inference
    for images, labels, names in tqdm(test_dataloader):
        #print(type(names))
        images = torch.tensor(images, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        # Forward pass
        outputs = model(images)
        # print(type(outputs))
        # print(outputs.shape)
        probabilities = torch.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)   # Convert to binary (0 or 1)
        pred = pred.to(device=device) #dtype=torch.float32
        maxvals, pred_index = torch.max(outputs, 1)
        # Collect labels and predictions
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
        all_names.extend(names)
        all_probabilities.extend(probabilities.cpu().numpy())
        
# 4. Compute accuracy or other evaluation metrics (e.g., confusion matrix)
# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.squeeze(np.array(all_preds))
all_names = np.array(all_names)
all_probabilities = np.array(all_probabilities)


# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_names = np.array(all_names)



if class_for_gradcam == 'merger':
    targetclass = 0
else:
    targetclass = 1
    
print('targetclass', targetclass)
#save_dir = 'gradcam/' + CNNName + '/test/predicted_label_' + class_for_gradcam 
save_dir = '/n/home09/aschechter/code/BinaryCNNTesting/PytorchCNNs/gradcam/zoobot/bothlabels'
if os.path.exists(save_dir):
    print('yes this directory exists')

#### GradCAM w
# We have to specify the target we want to generate the CAM for.
# targets = [ClassifierOutputTarget(0)] #target class is mergers
# print(targets)
#targets = [ClassifierOutputTarget(targetclass)] #target class is mergers
targets = [0, 1]
# print(targets)
print('ClassifierOutputTarget', ClassifierOutputTarget(targetclass), type(ClassifierOutputTarget(targetclass)))
path_test = []
targets_GradCAM = [model.encoder.layer4[1].conv2]
layer_names_GradCAM = ['layer4_conv2']
# Select the first validation image to visualize Grad-CAM
sample_index_choices = np.arange(0, len(test_dataloader.dataset))#len(test_dataloader.dataset))#  # Change this index to visualize different images
for sample_index in sample_index_choices:
    sample_image = test_dataloader.dataset[sample_index][0].unsqueeze(0)  # Add batch dimension
    sample_label = test_dataloader.dataset[sample_index][1]
    sample_name = test_dataloader.dataset[sample_index][2]
    
    # Move to device
    sample_image = sample_image.to(torch.float32).to(device)

    #make labels words
    if sample_label == 1.:
        sample_label_word = 'nonmerger'
    else:
        sample_label_word = 'merger'
        
    
    # Get the original image for visualization
    
    original_img = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
    predicted_class = torch.argmax((model(sample_image)), dim=1) 
    predicted_label = f"Predicted: {predicted_class}, True: {sample_label_word}"
    
    if predicted_class == 1.:
        predicted_class_word = 'nonmerger'
    else:
        predicted_class_word = 'merger'
        
    fig, axes = plt.subplots(1, len(targets)+1, figsize=(15, 6))  # Create subplots for each target class and original image
    axes = axes.ravel()  # Flatten axes array if it's multi-dimensional
    plt.subplots_adjust(wspace=0, hspace=0)
    # Original image subplot
    ax = axes[0]
    im_sum = np.sum(original_img, axis=2)
    mean = np.mean(im_sum)
    std = np.std(im_sum)
    contour_levels2 = [mean + i * std for i in [1,3,5]]
    im_original = ax.imshow(im_sum, cmap='magma', norm=LogNorm())  ##, extent = extent Log scale #vmin=np.min(im_sum)*1e-7, vmax=1
    ax.contour(im_sum, colors = 'white', levels = contour_levels2, zorder = 1)
    ax.set_title(f"Original Image - True: {sample_label_word}", fontsize = 'medium')
    ax.axis('off')
    counter = 1 #gradCAMS start in second subplot
    for target_class in targets:
        if target_class == 0:
            target_class_word = 'merger'    
        else:
            target_class_word = 'nonmerger'
        target = [ClassifierOutputTarget(target_class)] #setting to merger or nonmerger
        target_layer_GradCAM = targets_GradCAM[0]

        with GradCAM(model=model, target_layers=[target_layer_GradCAM]) as cam:
            grayscale_cam = cam(input_tensor=sample_image, targets=target)
            print(np.shape(grayscale_cam)) 
            grayscale_cam = grayscale_cam[0, :]  # Get the CAM for the first image in the batch
        # Convert the input image to a format suitable for visualization
        rgb_img = sample_image.squeeze().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
       
        # Visualize the Grad-CAM output
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)



    

        # Plot the Grad-CAM
        ax = axes[counter]
        im = ax.imshow(visualization) #, extent = extent
        ax.contour(im_sum, colors = 'white', levels = contour_levels2, zorder=1) #, extent = extent
        ax.set_title(f"Class Activated: {target_class_word} - Pred: {predicted_class_word}", fontsize = 'medium')
        ax.axis('off')

        counter += 1

    plt.tight_layout()
    plt.savefig(f"{save_dir}/gradcam_test_image_{sample_name}.png") #, dpi=300
    #plt.show()
    plt.clf()
    plt.close()
