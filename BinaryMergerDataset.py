
import numpy as np
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import glob
from astropy.visualization import simple_norm, SqrtStretch
from skimage.transform import resize
from SetRandomSeed import set_random_seeds




set_random_seeds(626)

# In[4]:


path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1widebinmocks/'
#stretch = AsinhStretch()
pad_val = 0
BATCH_SIZE = 64
NUM_EPOCHS = 40
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# In[5]:


torch.cuda.is_available()


# In[6]:


class BinaryMergerDataset(Dataset): #in future: put this in one file and always call it!
    def __init__(self, data_path, dataset, mergers = True, transform=None, codetest=True):
        self.dataset = dataset
        self.mergers = mergers
        self.codetest=codetest
        if self.dataset == 'train':
            if mergers == True:
                self.images = glob.glob(data_path + 'training/anymergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'training/anymergers/mergerlabel.npy')
                self.image_filenames = self.images
                #print('length of file list', len(self.images))
            else:
                self.images = glob.glob(data_path + 'training/nonmergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'training/nonmergers/mergerlabel.npy')
                self.image_filenames = self.images
                #print('length of file list', len(self.images))
        elif self.dataset == 'validation':
            if mergers == True:
                self.images = glob.glob(data_path + 'validation/anymergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'validation/anymergers/mergerlabel.npy')
                self.image_filenames = self.images
            else:
                self.images = glob.glob(data_path + 'validation/nonmergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'validation/nonmergers/mergerlabel.npy')
                self.image_filenames = self.images
        elif self.dataset == 'test':
            if mergers == True:
                self.images = glob.glob(data_path + 'test/anymergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'test/anymergers/mergerlabel.npy')
                self.image_filenames = self.images
            else:
                self.images = glob.glob(data_path + 'test/nonmergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'test/nonmergers/mergerlabel.npy')
                self.image_filenames = self.images
        
        self.transform = transform
        

    def __len__(self):
        if self.codetest:
            return len(self.img_labels[0:200])
        else:   
            return len(self.img_labels)

    def __getitem__(self, idx):
        #print(np.shape(self.images))
        img_path = self.images[idx]
        img_name = self.image_filenames[idx]
        #print(idx)
        image = np.load(img_path) #keep as np array to normalize
        #print('first shape', np.shape(image))
        #image = np.transpose(image, (2,1,0))
        #print('image shape', np.shape(image))
        image1 = image[:,:,0:3]
        #print('necesary image shape: ', np.shape(image1))
        image = image[:, :, [0, 1, 3]]
        #image = image[:,:,[[0,2,3]]][:,:,0,:] #had to play around with indices to get rid of 606 filter
        #print('image shape: ', np.shape(image))
        #image = stretch(image)
        # ##clip values to get rid of outliers -- np.clip makes the percentiles the new min and max
        # global_min = np.percentile(image, 1)
        # global_max = np.percentile(image, 99)
        # image = np.clip(image, a_min=global_min, a_max=global_max)
        #resize to 224 for resnet
        image = resize(image, (224,224))
        #print('image shape: ', np.shape(image))
        image = image * 1e20 #test to get magnitudes up
        # power = simple_norm(image, 'power', power = 0.5)
        # image = power(image)
        norm = simple_norm(image, 'log', log_a=2e5)
        image = norm(image)
        label_file = self.img_labels
        #print('label shape', np.shape(label_file))
        label = label_file[idx]
        #print('first label call: ', np.shape(label))
        # if label != 0:
        #     print(label)
        #print(labels)
        #label = np.load(label_path)[idx]
        #print('label shape: ',np.shape(labels))
        # split word
        w = 'allfilters'
        _, _, name = img_name.partition(w)
        name = name[:-4]
        # print(name)
        # print(name[:-4])
        
        if self.transform is not None:
            image = self.transform(image)

        return image, int(label), name


# In[7]:


def get_transforms(aug=True):
    transforms = []
    transforms.append(T.ToTensor())
    #transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if aug == True:
        transforms.extend([
        #transforms.append(torch.nn.Sequential(
        T.RandomRotation(90), 
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5)
        #T.Pad(pad_val), #this line added for resnet
        ])
    else: transforms.append(T.Pad(pad_val))
        
    return T.Compose(transforms)

