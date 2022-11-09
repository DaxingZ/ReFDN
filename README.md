# ReFDN

Official Pytorch implementation of the paper "Lightweight Remote-Sensing Image Super-Resolution via Re-parameterized Feature Distillation Network". The code will release when the paper is accepted.

# How to evaluate

Evaluate the reconstruction accuracy in terms of PSNR/SSIM, where the path of SR/HR images could be edited in 
```
cd metric_scripts 
python calculate_PSNR_SSIM.py
```
Evaluate the efficiency of models in terms of parameters, FLOPs, inference time, and so on, where the input settings could be edited in
```
cd metric_scripts 
python calculate_Efficiency.py
```
