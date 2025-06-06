
###############################  NORMAL  ##########################################
## 1000

## CIFAR10
CUDA_VISIBLE_DEVICES=3 nohup python -u STBA_Freq.py -D CIFAR-10 > logs_Freq/clean/CIFAR-10/STBA_q1000.log  2>&1 & # Kunpeng 37129

## CIFAR100
CUDA_VISIBLE_DEVICES=2 nohup python -u STBA_Freq.py -D CIFAR-100 > logs_Freq/clean/CIFAR-100/STBA_q1000.log  2>&1 & # Kunpeng 37746

## STL-10
CUDA_VISIBLE_DEVICES=1 nohup python -u STBA_Freq.py -D STL-10 > logs_Freq/clean/STL-10/STBA_q1000.log  2>&1 & #  Kunpeng 39943

## ImageNet
CUDA_VISIBLE_DEVICES=3 nohup python -u STBA_Freq.py -D ImageNet > logs_Freq/clean/ImageNet/STBA_q1000.log  2>&1 & # Kunpeng 43480


## 10000
## CIFAR10
CUDA_VISIBLE_DEVICES=0 nohup python -u STBA_Freq.py -D CIFAR-10 -Q 10000 > logs_Freq/clean/CIFAR-10/STBA_q10000.log  2>&1 & # Kunpeng 45554

## CIFAR100
CUDA_VISIBLE_DEVICES=1 nohup python -u STBA_Freq.py -D CIFAR-100 -Q 10000 > logs_Freq/clean/CIFAR-100/STBA_q10000.log  2>&1 & #  Kunpeng 45687

## STL-10
CUDA_VISIBLE_DEVICES=2 nohup python -u STBA_Freq.py -D STL-10 -Q 10000 > logs_Freq/clean/STL-10/STBA_q10000.log  2>&1 & #  Kunpeng 45827

## ImageNet 
CUDA_VISIBLE_DEVICES=3 nohup python -u STBA_Freq.py -D ImageNet -Q 10000 > logs_Freq/clean/ImageNet/STBA_q10000.log  2>&1 & #  Kunpeng 45975



## ## TEST
CUDA_VISIBLE_DEVICES=1 python -u STBA_Freq.py -D STL-10


###############################  ROBUST  ##########################################

## 1000
## CIFAR10
CUDA_VISIBLE_DEVICES=0 nohup python -u STBA_Freq.py -D CIFAR-10 -T defense > logs_Freq/robust/CIFAR-10/STBA_q1000.log  2>&1 & # Fenghuang 646521

## CIFAR100
CUDA_VISIBLE_DEVICES=2 nohup python -u STBA_Freq.py -D CIFAR-100 -T defense >> logs_Freq/robust/CIFAR-100/STBA_q1000.log  2>&1 & # Wuma  

## STL-10
CUDA_VISIBLE_DEVICES=1 nohup python -u STBA_Freq.py -D STL-10 -T defense > logs_Freq/robust/STL-10/STBA_q1000.log  2>&1 & #  Kengpeng 55986

## ImageNet
CUDA_VISIBLE_DEVICES=2 nohup python -u STBA_Freq.py -D ImageNet -T defense > logs_Freq/robust/ImageNet/STBA_q1000.log  2>&1 & # Kengpeng 10888


## 10000

## CIFAR10
CUDA_VISIBLE_DEVICES=1 nohup python -u STBA_Freq.py -D CIFAR-10 -Q 10000 -T defense > logs_Freq/robust/CIFAR-10/STBA_q10000.log  2>&1 & # Fenghuang 646610

## CIFAR100
CUDA_VISIBLE_DEVICES=3 nohup python -u STBA_Freq.py -D CIFAR-100 -Q 10000 -T defense >> logs_Freq/robust/CIFAR-100/STBA_q10000.log  2>&1 & #  Kengpeng 11874

## STL-10
CUDA_VISIBLE_DEVICES=3 nohup python -u STBA_Freq.py -D STL-10 -Q 10000 -T defense > logs_Freq/robust/STL-10/STBA_q10000.log  2>&1 & #  Kengpeng 2831

## ImageNet 
CUDA_VISIBLE_DEVICES=1 nohup python -u STBA_Freq.py -D ImageNet -Q 10000 -T defense > logs_Freq/robust/ImageNet/STBA_q10000.log  2>&1 & #  Kengpeng 11017

## ## TEST

CUDA_VISIBLE_DEVICES=1 python -u STBA_Freq.py -D ImageNet 