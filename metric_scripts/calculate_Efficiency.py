import torch
import torch.nn as nn
import time
from test_summary import get_model_flops, get_model_activation
from model import ReFDN
if __name__ == "__main__":
    model =ReFDN()
    device = torch.device("cuda:0")
    model.to(device)
    input_dim = (3,64,64)  # set the input dimension
    ==== #Conv2d ====
    num_conv = get_model_activation(model, input_dim)
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))
    
    ==== #FLOPs ====
    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))
    
    ==== #Params ====
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
    
    ==== #Inference Time ====
    iterations = 200 
    random_input = torch.randn(1, 3, 256, 256).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # GPU warm-up
    for _ in range(50):
        _ = model(random_input)
    # testing
    times = torch.zeros(iterations)    
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(random_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) 
            times[iter] = curr_time
    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
