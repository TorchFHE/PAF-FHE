# PAF-FHE: Low-Cost Accurate Non-Polynomial Operator Polynomial Approximation in Fully Homomorphic Encryption Based ML Inference [[Paper]](https://assets.researchsquare.com/files/rs-2910088/v1_covered_0e6c94bd-1499-4b2c-b414-902c232b490c.pdf?c=1683777877) [[Poster]](https://jianmingtong.github.io/publications/PAF_FHE_poster.pdf)
```
@misc {PPR:PPR658940, 
    Title = {PAF-FHE: Low-Cost Accurate Non-Polynomial Operator Polynomial Approximation in Fully Homomorphic Encryption Based ML Inference}, Author = {Dang, Jingtian and Tong, Jianming and Golder, Anupam and Raychowdhury, Arijit and Hao, Cong and Krishna, Tushar}, 
    DOI = {10.21203/rs.3.rs-2910088/v1}, 
    Publisher = {Research Square}, 
    Year = {2023}, 
    URL = {https://doi.org/10.21203/rs.3.rs-2910088/v1}, }
```
## Secure Fully Homomorphic Encryption (FHE) based Machine Learning Inference Converts Non-polynomial Operators (ReLU/MaxPooling) into Polynomial Approximation Functions (PAF)
![](image/secure_ML_inference.png)

## Existing PAFs suffer from either prohibitive latency overhead or low accuracy. PAF-FHE proposes four training techniques to enable exploration on the entire PAF degree space and spot high-accuracy low-latency PAF.
![](image/RelatedWork.png)

# Ready to run?

```
#Activate Conda
# Create a python3.8 enviroment
conda create --name PAF-FHE  python=3.8

# Activate the enviroment
conda activate PAF-FHE

# Install package
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge pytorch-lightning

# Download cifar10 pretrained models
cd PyTorch_CIFAR10/
sh download_weights.sh 
cd ..

# Open /global_config/global_config.yaml
#Edit "dataset_dirctory:" to set a folder to store dataset.

# Download dataset
python3 util.py -dd True --dataset cifar10
python3 util.py -dd True --dataset cifar100
python3 util.py -dd True --dataset imagenet
```


## Control Parameters for the library
```
typical step
For one model with a dataset, one -wd (working directory) should be used
--model: 		resnet18, vgg19_bn, resnet32
--dataset: 	cifar10, imagenet, cifar100
-st: 			a7, 2f12g1, f2f2, f2g3, f1g2
Supported combination: vgg19_bn & imagenet, vgg19_bn & cifar10, resnet18 & imagenet, and resnet32 & cifar100
-st is the supported PAF type
-dc stands for "data collection": 
```

## ResNet-18 on ImageNet_1k
```
# The following steps must be run in serial, as following steps need results from previous steps.
# Collection CT data
python3 ./CT.py --model resnet18 --dataset imagenet_1k -wd ../resnet18_imagenet1k/ -dc True
# CT
python3 ./CT.py --model resnet18 --dataset imagenet_1k -wd ../resnet18_imagenet1k/ -st 2f12g1
# PA and AT
python3 ./PA_AT.py --model resnet18 --dataset imagenet_1k -wd ../resnet18_imagenet1k/ -st 2f12g1
# Statistic Scale.
python3 ./SS.py --model resnet18 --dataset imagenet_1k -wd ../resnet18_imagenet1k/ -st 2f12g1
```
## ResNet-32 on CiFar-100
```
# The following steps must be run in serial, as following steps need results from previous steps.
# Collection CT data
python3 ./CT.py --model resnet32 --dataset cifar100 -wd ../resnet32_cifar100/ -dc True
# CT
python3 ./CT.py --model resnet32 --dataset cifar100 -wd ../resnet32_cifar100/ -st 2f12g1
# PA and AT
python3 ./PA_AT.py --model resnet32 --dataset cifar100 -wd ../resnet32_cifar100/ -st 2f12g1
# Statistic Scale.
python3 ./SS.py --model resnet32 --dataset cifar100 -wd ../resnet32_cifar100/ -st 2f12g1
```

## VGG-19 on CiFar-10
```
# The following steps must be run in serial, as following steps need results from previous steps.
# Collection CT data
python3 ./CT.py --model vgg19_bn --dataset cifar10 -wd ../vgg19_bn_cifar10/ -dc True
# CT
python3 ./CT.py --model vgg19_bn --dataset cifar10 -wd ../vgg19_bn_cifar10/ -st 2f12g1
# PA and AT
python3 ./PA_AT.py --model vgg19_bn --dataset cifar10 -wd ../vgg19_bn_cifar10/ -st 2f12g1
# Statistic Scale.
python3 ./SS.py --model vgg19_bn --dataset cifar10 -wd ../vgg19_bn_cifar10/ -st 2f12g1
```

## VGG-19 on ImageNet_1k
```
# The following steps must be run in serial, as following steps need results from previous steps.
# Collection CT data
python3 ./CT.py --model vgg19_bn --dataset imagenet_1k -wd ../vgg19_bn_imagenet1k/ -dc True
# CT
python3 ./CT.py --model vgg19_bn --dataset imagenet_1k -wd ../vgg19_bn_imagenet1k/ -st 2f12g1
# PA and AT
python3 ./PA_AT.py --model vgg19_bn --dataset imagenet_1k -wd ../vgg19_bn_imagenet1k/ -st 2f12g1
# Statistic Scale.
python3 ./SS.py --model vgg19_bn --dataset imagenet_1k -wd ../vgg19_imagenet1k/ -st 2f12g1
```
