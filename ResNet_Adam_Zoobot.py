
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import confusion_matrix
import glob
from astropy.visualization import simple_norm
from tqdm import tqdm
from skimage.transform import resize
from BinaryMergerDataset import BinaryMergerDataset, get_transforms
import pickle
from SetRandomSeed import set_random_seeds, GeneratorSeed
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier



set_random_seeds(626)
g = GeneratorSeed(626)
path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1widebinmocks/'
pad_val = 0
BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 64
NUM_EPOCHS = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)




train_mergers_dataset_augment = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(aug=True), codetest=False)
train_nonmergers_dataset_augment = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(aug=True), codetest=False)

train_mergers_dataset_orig = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(aug=False), codetest=False)
train_nonmergers_dataset_orig = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(aug=False), codetest=False)

train_dataset_full = torch.utils.data.ConcatDataset([train_mergers_dataset_augment, train_nonmergers_dataset_augment, train_mergers_dataset_orig, train_nonmergers_dataset_orig])
train_dataloader = DataLoader(train_dataset_full, shuffle = True, num_workers = 0, batch_size=BATCH_SIZE, generator=g)

validation_mergers_dataset_orig = BinaryMergerDataset(path, 'validation', mergers = True, transform = get_transforms(aug=False), codetest=False)
validation_nonmergers_dataset_orig = BinaryMergerDataset(path, 'validation', mergers = False, transform = get_transforms(aug=False), codetest=False)

validation_dataset_full = torch.utils.data.ConcatDataset([validation_mergers_dataset_orig, validation_nonmergers_dataset_orig])
indices = np.random.permutation(len(validation_dataset_full))
shuffled_validation_dataset = Subset(validation_dataset_full, indices)
validation_dataloader = DataLoader(validation_dataset_full, shuffle = False, num_workers = 0, batch_size=VALIDATION_BATCH_SIZE, generator = g)#num workers used to be 4




total_size_train = len(train_dataset_full)
total_size_validation = len(validation_dataset_full)

print(total_size_train)

number_of_iterations = total_size_train//BATCH_SIZE #// is floor
print('number_of_iterations', number_of_iterations)

learning_rate = 1e-5
lr_decay = 0.5


###https://www.kaggle.com/code/kvpratama/pretrained-resnet18-in-pytorch
model = FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-resnet18', learning_rate=learning_rate,  # use a low learning rate
    layer_decay=lr_decay,  # reduce the learning rate from lr to lr^0.5 for each block deeper in the network
    # arguments specific to FinetuneableZoobotClassifier
    num_classes=2
)



model = model.to(device)
#Check which layer in the model that will compute the gradient
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)


modelloss = {} #loss history
modelloss['train'] = []
modelloss['validation'] = []
modelacc = {} #used later for accuracy
modelacc['train'] = []
modelacc['validation'] = []
x_epoch = []

columndata = {}
columndata['accuracy_column'] = []
columndata['output_column'] = []
columndata['label_column'] = []




def get_accuracy_BCE(pred,original):
    pred = (torch.sigmoid(pred) > 0.5).float() #pred is now 0 or 1
    pred = pred.cpu().detach().numpy()
    original = original.cpu().numpy()

    return torch.mean(pred == original) * 100

def get_accuracy_CE(pred,original):
    return np.mean(pred.cpu().numpy() == original.long().cpu().numpy()) * 100

def plot_confusion_matrix(cm, classes, epoch): #help from chat GPT
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix After' + str(epoch) + 'Epochs')
    plt.savefig('ConfusionMatrix_ResNet18_Adam' + str(learning_rate) + '.png', dpi = 300)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

fig = plt.figure(figsize = (12, 6))
ax0 = fig.add_subplot(121, title="Loss")
ax1 = fig.add_subplot(122, title="Accuracy")
#plt.suptitle('lr = ' + str(learning_rate) + '| cyclic max = ' + str(max_learning_rate) + 'step size =' + str(step_size) + ' batch' + str(BATCH_SIZE))
plt.suptitle('lr = ' + str(learning_rate) + '| Cyclic '+ ' batch' + str(BATCH_SIZE))

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    # print('xshape', np.shape(x_epoch))
    # print('yshape', np.shape(modelloss['train']))
    ax0.plot(x_epoch, modelloss['train'], label='train', color = 'rebeccapurple')
    ax0.plot(x_epoch, modelloss['validation'], label='val', color = 'darkorange')
    ax1.plot(x_epoch, modelacc['train'], label='train', color = 'rebeccapurple')
    ax1.plot(x_epoch, modelacc['validation'],  label='val', color = 'darkorange')
    ax1.axhline(y= 65, color = 'grey', linestyle = '--')
    ax1.axhline(y= 75, color = 'grey', linestyle = '--')
    if np.max(modelloss['validation'])> 5:
        ax0.set_ylim(0, 5)
    else:
        ax0.set_ylim(np.min(modelloss['validation']) - 0.05,np.max(modelloss['validation']) + 0.05)
    ax1.set_ylim(50,100)
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    plt.tight_layout()
    fig.savefig('metrics_zoobot_Adam' + str(learning_rate) + '.png')


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float(np.inf)
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        # If this is the first validation loss value or if it's better than the best we've seen
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:  # No improvement
            self.counter += 1
            if self.counter >= self.patience:  # If we've reached the patience, stop early
                self.early_stop = True
                print('Early Stopped!')
                

            

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

best_val_loss = float(np.inf)  # Initialize to a very large value
best_model_state = None
best_epoch = -1
        
earlystop = EarlyStopping(patience=5, delta = 0.0005)        
for epoch in range(NUM_EPOCHS):
    t_epoch_loss = 0.0
    v_epoch_loss = 0.0
    t_accuracy = []
    v_accuracy = []
    t_counter = 0
    v_counter = 0

    for images, labels, names in tqdm(iter(train_dataloader)):
        model.train(True) #default is not in training mode - need to tell pytorch to train
        bs = images.shape[0]         #batch size
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1) #single prediction per image of class with highest probability
        maxvals, pred_index = torch.max(outputs, 1) #preds = raw logits; index of max for back prop
        for o in range(len(outputs)):
            columndata['output_column'].append(pred[o].item())
            columndata['label_column'].append(labels[o].item())

        pred = pred.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)

        loss = F.cross_entropy(outputs, labels.long())
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        t_epoch_loss += loss.item() * bs #make the loss a number
        #print('pred', type(pred))
        t_accuracy.append(get_accuracy_CE(pred, labels))
        t_counter +=1 

    #look at outputs here and what shape I want!   
    modelacc['train'].append(np.mean(np.array(t_accuracy)))

    #print('loss shape', np.shape(loss))
    modelloss['train'].append(t_epoch_loss/total_size_train)

    print('TRAINING LOSS: ', t_epoch_loss)

    model.eval()
    all_labels = []
    all_preds = []
    all_features = [] #for t-SNE
    saliency_maps = []
    #with torch.no_grad():
    for images, labels, names in tqdm(iter(validation_dataloader)):
        #model.train(False)
        bs = images.shape[0] 

        images = torch.tensor(images, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)

        # Set requires_grad to True for saliency map calculation ##help from chatGPT
        images.requires_grad_()

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1) 
        maxvals, pred_index = torch.max(outputs, 1) #help from chat GPT maxvals sort which class it belongs to but we only want preds
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs.detach().cpu().numpy())
        all_features.extend(pred.detach().cpu().numpy())
        labels = labels.to(device=device, dtype=torch.float32)
        loss = F.cross_entropy(outputs, labels.long())
        v_epoch_loss += loss.item() * bs
        v_accuracy.append(get_accuracy_CE(pred, labels))
        v_counter += 1
        
        model.zero_grad()
        gradients_array = torch.zeros_like(outputs)
        gradients_array[:, 0] = 1  # Assuming class 0 is at index 0
        outputs.backward(gradient=gradients_array)     
    modelacc['validation'].append(np.mean(np.array(v_accuracy)))

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_features = np.array(all_features)
    all_preds = np.rint(all_preds)

    modelloss['validation'].append(v_epoch_loss/total_size_validation)
    if v_epoch_loss/total_size_validation < best_val_loss:
        best_val_loss = v_epoch_loss/total_size_validation
        best_model_state = model.state_dict()
        best_epoch = epoch

    #want to draw curve and then stop the loop
    draw_curve(epoch)
    earlystop(val_loss=v_epoch_loss/total_size_validation, model=model)
    if earlystop.early_stop:
        break
    
##save for reloading and plots
loss_file = open('Adam_Cyclic_loss', 'wb') 
pickle.dump(modelloss, loss_file) 
loss_file.close() 

acc_file = open('Adam_Cyclic_acc', 'wb') 
pickle.dump(modelacc, acc_file) 
acc_file.close() 
#torch.save(model.state_dict(), 'model_weights.pth')
torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(), 
            'loss': best_val_loss,
            }, 'ResNet_Adam_Zoobot.pth')

#CONFUSION MATRIX
print('allfeatures', np.shape(all_features))
print('alllabels', np.shape(all_labels))
cm = confusion_matrix(all_labels, all_features)
plot_confusion_matrix(cm, ['Merger', 'Nonmerger'], epoch)





