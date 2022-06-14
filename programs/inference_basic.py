import os
import torch
import math
import torch.nn.functional as F
from src.neurips_bdl_starter_kit import pytorch_models as p_models


# directories
dir_data = os.path.join(os.path.dirname(__file__), '../data')
dir_processing = os.path.join(dir_data, 'processing')
dir_cifar_clean = os.path.join(dir_processing, 'cifar')
# filenames
# # raw cifar 
# dir_cifar_bdl = os.path.join(dir_data, 'cifar10_bdl_comp')
# dir_cifar10_1 = os.path.join(dir_data, 'cifar10.1')
# # dataset filenames
# filepath_dataset_train = os.path.join(dir_cifar_clean, 'dataset_cifar_train.pt')
# filepath_dataset_test0 = os.path.join(dir_cifar_clean, 'dataset_cifar_test0.pt')
# filepath_dataset_test1 = os.path.join(dir_cifar_clean, 'dataset_cifar_test1.pt')
# loader filenames
filepath_loader_train = os.path.join(dir_cifar_clean, 'loader_cifar_train.pt')
filepath_loader_test0 = os.path.join(dir_cifar_clean, 'loader_cifar_test0.pt')
filepath_loader_test1 = os.path.join(dir_cifar_clean, 'loader_cifar_test1.pt')

BATCH_SIZE = 100 # MUST BE 100 TO MATCH STUDY
BATCH_SIZE_TEST = 100 # MUST BE 100 TO MATCH STUDY


def main(model_key, prior_variance, str_device='mps'):
    # device config
    device = torch.device(str_device)

    # Get model
    net_fn =  p_models.get_model(model_key, data_info={"num_classes": 10})
    if torch.cuda.is_available():
        print("GPU available!")
        net_fn = net_fn.cuda()
    else: 
        net_fn = net_fn.to(device)

    # define likelihood function w.r.t. 'batch'
    def log_likelihood_fn(model_state_dict, batch):
        """Computes the log-likelihood.
        TODO - extend to 
        """
        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        else:
            x = x.to(device)
            y = y.to(device)

        net_fn.zero_grad()

        for name, param in net_fn.named_parameters():
            param.data = model_state_dict[name]
        
        logits = net_fn(x)
        num_classes = logits.shape[-1]
        labels = F.one_hot(y.to(torch.int64), num_classes= num_classes)
        softmax_xent = torch.sum(labels * F.log_softmax(logits))

        return softmax_xent

    def log_prior_fn(model_state_dict):
        """Computes the Gaussian prior log-density."""
        n_params = sum(p.numel() for p in model_state_dict.values()) 
        exp_term = sum((-p**2 / (2 * prior_variance)).sum() for p in model_state_dict.values() )
        norm_constant = -0.5 * n_params * math.log((2 * math.pi * prior_variance))
        return exp_term + norm_constant

    def log_posterior_fn(model_state_dict, batch):
        log_lik = log_likelihood_fn(model_state_dict, batch)
        log_prior = log_prior_fn(model_state_dict)
        return log_lik + log_prior

    def get_accuracy_fn(net_fn, batch, model_state_dict):
        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        else: 
            x = x.to(device)
            y = y.to(device)
        
        # get logits 
        net_fn.eval()
        with torch.no_grad():
            for name, param in net_fn.named_parameters():
                param.data = model_state_dict[name]
        logits = net_fn(x)
        net_fn.train()

        # get log probs 
        log_probs = F.log_softmax(logits, dim=1)
        # get preds 
        probs = torch.exp(log_probs)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean()
        
        return accuracy, probs

    def evaluate_fn(data_loader, model_state_dict):
        sum_accuracy = 0
        all_probs = []
        for x, y in data_loader:       
            batch_accuracy, batch_probs = get_accuracy_fn((x, y), model_state_dict)
            sum_accuracy += batch_accuracy.item()
            all_probs.append(batch_probs)
        all_probs = torch.cat(all_probs, dim=0)
        return sum_accuracy / len(data_loader), all_probs

    batch_size = BATCH_SIZE
    test_batch_size = BATCH_SIZE_TEST
    num_epochs = 10
    momentum_decay = 0.9
    lr = 0.001
    
    loader_train = torch.load(filepath_loader_train)
    loader_test0 = torch.load(filepath_loader_test0)
    loader_test1 = torch.load(filepath_loader_test1)



if __name__ == '__main__':
    
    # model_key = "cifar_alexnet"
    prior_variance = 5.
    model_key = "resnet20_frn_swish"

    main(model_key, prior_variance)