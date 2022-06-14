#!/bin/bash
# get CIFAR-10.1 dataset, ref Recht, B. et al. (2018)

# get data for CIFAR-10.1 (benign shift)
git submodule add --force https://github.com/modestyachts/CIFAR-10.1.git

# move up one folder
cd ..

# move data to new folder and then delete repo
mkdir data/cifar10.1
mv src/CIFAR-10.1/datasets/* data/cifar10.1
rm -r src/CIFAR-10.1
git rm --cached src/CIFAR-10.1

# end
