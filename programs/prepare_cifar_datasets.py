import os
from pathlib import Path
import requests

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# global variables...

# directories
dir_data = os.path.join(os.path.dirname(__file__), '../data')
dir_raw = os.path.join(dir_data, 'raw')
dir_processing = os.path.join(dir_data, 'processing')
dir_cifar_clean = os.path.join(dir_processing, 'cifar')
# filenames
# raw cifar 
dir_cifar_bdl = os.path.join(dir_raw, 'cifar10_bdl_comp')
dir_cifar10_1 = os.path.join(dir_raw, 'cifar10.1')
# dataset filenames
filepath_dataset_train = os.path.join(dir_cifar_clean, 'dataset_cifar_train.pt')
filepath_dataset_test0 = os.path.join(dir_cifar_clean, 'dataset_cifar_test0.pt')
filepath_dataset_test1 = os.path.join(dir_cifar_clean, 'dataset_cifar_test1.pt')
# loader filenames
filepath_loader_train = os.path.join(dir_cifar_clean, 'loader_cifar_train.pt')
filepath_loader_test0 = os.path.join(dir_cifar_clean, 'loader_cifar_test0.pt')
filepath_loader_test1 = os.path.join(dir_cifar_clean, 'loader_cifar_test1.pt')

BATCH_SIZE = 100 # MUST BE 100 TO MATCH STUDY
BATCH_SIZE_TEST = 100 # MUST BE 100 TO MATCH STUDY

# for cifar10.1
cifar10_label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 

def download_raw_cifar_bdl(_dir):
    
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    
    # # Get (raw) Data for competition (must match exactly)
    dict_url_cifar10 = {
        'cifar10_train_x.csv': 'https://storage.googleapis.com/neurips2021_bdl_competition/cifar10_train_x.csv',
        'cifar10_train_y.csv': 'https://storage.googleapis.com/neurips2021_bdl_competition/cifar10_train_y.csv',
        'cifar10_test_x.csv': 'https://storage.googleapis.com/neurips2021_bdl_competition/cifar10_test_x.csv',
        'cifar10_test_y.csv': 'https://storage.googleapis.com/neurips2021_bdl_competition/cifar10_test_y.csv'
    }

    # Manually download with these links:
    for f, url in dict_url_cifar10.items():
        response = requests.get(url)
        f_path = os.path.join(_dir, f)
        # save raw to disk
        open(f_path, 'wb').write(response.content)
        assert os.path.exists(f_path)
    
    return None

def get_cifar_bdl(_dir):

    # np array
    x_train = np.loadtxt(os.path.join(_dir, "cifar10_train_x.csv"))
    y_train = np.loadtxt(os.path.join(_dir, "cifar10_train_y.csv"))
    x_test = np.loadtxt(os.path.join(_dir, "cifar10_test_x.csv"))
    y_test = np.loadtxt(os.path.join(_dir, "cifar10_test_y.csv"))

    # reshape
    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    x_test = x_test.reshape((len(x_test), 3, 32, 32))

    trainset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    testset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

    return trainset, testset

def get_cifar_10_1_v6(_dir):
    """adapted from Richt, B. et al. (2018) github CIFAR10.1.

    Args:
        _dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    version_string = 'v6'
    filename_labels = f'cifar10.1_{version_string}_labels.npy'
    filename_imagedata = f'cifar10.1_{version_string}_data.npy'
    filepath_labels = os.path.join(_dir, filename_labels)
    filepath_imagedata =  os.path.join(_dir, filename_imagedata)

    # load labels
    assert Path(filepath_labels).is_file()
    y_test = np.load(filepath_labels)

    # load images
    assert Path(filepath_imagedata).is_file()
    x_test = np.load(filepath_imagedata)

    # basic checks... 
    assert len(y_test.shape) == 1
    assert len(x_test.shape) == 4
    assert y_test.shape[0] == x_test.shape[0]
    assert x_test.shape[1] == 32
    assert x_test.shape[2] == 32
    assert x_test.shape[3] == 3

    if version_string == 'v6' or version_string == 'v7':
        assert y_test.shape[0] == 2000

    x_test = x_test.reshape((len(x_test), 3, 32, 32)) # shap matches cifar10_bdl...

    testset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    
    return testset


def main():
    """get raw cifar (bdl comp, cifar10.1).

    create dataset and loader objects. 
    
    Save to disk.

    Raises:
        e: _description_

    Returns:
        _type_: _description_
    """
    # if raw data not downloaded... 
    if not os.path.exists(os.path.join(dir_cifar_bdl, 'cifar10_test_x.csv')):
        download_raw_cifar_bdl(dir_cifar_bdl)

    # upload and prep CIFAR-10 (from hmc baseline)
    dataset_train, dataset_test0 = get_cifar_bdl(dir_cifar_bdl)

    try:
        dataset_test1 = get_cifar_10_1_v6(dir_cifar10_1)
    except FileNotFoundError as e:
        print("Navigate to 'src' folder and run bash script './get_new_cifar10_dataset.sh'")
        print()
        raise e
    
    # save to 'dataloader'
    

    # cifar train and test
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_test0 = DataLoader(dataset_test0, batch_size=BATCH_SIZE_TEST, shuffle=False) 
    # new cifar10-1 (with benign shift)
    loader_test1 = DataLoader(dataset_test1, batch_size=BATCH_SIZE_TEST, shuffle=False)

    # save out datasets 
    

    # make dir
    if not os.path.exists(dir_cifar_clean):
        os.makedirs(dir_cifar_clean)
    
    # save datasets
    torch.save(dataset_train, filepath_dataset_train)
    torch.save(dataset_test0, filepath_dataset_test0)
    torch.save(dataset_test1, filepath_dataset_test1)
    # save loaders
    torch.save(loader_train, filepath_loader_train)
    torch.save(loader_test0, filepath_loader_test0)
    torch.save(loader_test1, filepath_loader_test1)

    # delete scraps

    return None

if __name__ == '__main__':
    main()