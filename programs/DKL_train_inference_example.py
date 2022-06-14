from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR    
import torch.nn.functional as F
from torch import nn
import torch
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import gpytorch
import math
import tqdm

import sys

# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'

repo_root = os.path.join(os.getcwd())
assert os.getcwd().split('/')[-1] == 'uncertainty_survey', "interpreter not in correct position, or src files need to package..."
sys.path.append(repo_root)

_device_ = torch.device('mps')

from src.models.DKL_trial import DenseNetFeatureExtractor, DKLModel

def main(dataset='cifar10'):

    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    aug_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    common_trans = [transforms.ToTensor(), normalize]
    train_compose = transforms.Compose(aug_trans + common_trans)
    test_compose = transforms.Compose(common_trans)

    if ('CI' in os.environ):  # this is for running the notebook in our testing framework
        train_set = torch.utils.data.TensorDataset(torch.randn(8, 3, 32, 32), torch.rand(8).round().long())
        test_set = torch.utils.data.TensorDataset(torch.randn(4, 3, 32, 32), torch.rand(4).round().long())
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=False)
        num_classes = 2
    elif dataset == 'cifar10':
        train_set = dset.CIFAR10('data', train=True, transform=train_compose, download=True)
        test_set = dset.CIFAR10('data', train=False, transform=test_compose)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
        num_classes = 10
    elif dataset == 'cifar100':
        train_set = dset.CIFAR100('data', train=True, transform=train_compose, download=True)
        test_set = dset.CIFAR100('data', train=False, transform=test_compose)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
        num_classes = 100
    else:
        raise RuntimeError('dataset must be one of "cifar100" or "cifar10"')

    # get feature extractor
    feature_extractor = DenseNetFeatureExtractor(block_config=(5, 5, 5), num_classes=num_classes, drop_rate=.1)
    num_features = feature_extractor.classifier.in_features

    model = DKLModel(feature_extractor, num_dim=num_features)
    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes)

    model = model.to(_device_)
    likelihood = likelihood.to(_device_)
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     likelihood = likelihood.cuda()

    
    ########################################
    ############ TRAINING SETUP ############
    ########################################
    n_epochs = 1
    lr = 0.1
    optimizer = SGD([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
        {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
        {'params': model.gp_layer.variational_parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
    scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

    def train(epoch):
        """_summary_
        """
        model.train()
        likelihood.train()

        minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        with gpytorch.settings.num_likelihood_samples(8):
            for data, target in minibatch_iter:
                
                if torch.cuda.is_available():
                    data, target = data.to(_device_), target.to(_device_)
                else:
                    try: 
                        data, target = data.to(_device_), target.to(_device_)
                    except:
                        raise Exception("Can't test on cpu or mps")
                
                optimizer.zero_grad()
                output = model(data)
                loss = -mll(output, target)
                loss.backward()
                optimizer.step()
                minibatch_iter.set_postfix(
                    loss=loss.item()
                )
    
    def test():
        """_summary_
        """
        model.eval()
        likelihood.eval()

        correct = 0
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                else:
                    try: 
                        data, target = data.to(_device_), target.to(_device_)
                    except:
                        raise Exception("Can't test on cpu or mps")
                
                output = likelihood(model(data))  # This gives us 16 samples from the predictive distribution
                pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
                correct += pred.eq(target.view_as(pred)).cpu().sum()
        print('Test set: Accuracy: {}/{} ({}%)'.format(
            correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
        ))

    for epoch in range(1, n_epochs + 1):
        # Whether or not to use Toeplitz math with gridded data, grid inducing point modules 
        # Pros: memory efficient, faster on CPU Cons: slower on GPUs with < 10000 inducing points
        with gpytorch.settings.use_toeplitz(False):
            with gpytorch.settings.cholesky_jitter(1e-3):
                train(epoch)
                test()
        
        scheduler.step()
        state_dict = model.state_dict()
        likelihood_state_dict = likelihood.state_dict()

        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, 'data/results/dkl_cifar_checkpoint.dat')
    
    return None



if __name__ == '__main__':
    main()