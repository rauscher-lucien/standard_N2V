# benchmark_pytorch.py
import sys
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torchvision
import torch.utils.benchmark as benchmark

from network import Noise2NoiseUNet3D, init_weights

import logging
logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log_file = open('logfile.log', 'w', buffering=1)
sys.stdout = log_file
sys.stderr = log_file


@torch.no_grad()
def measure_time_host(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_repeats: int = 100,
    num_warmups: int = 10,
    synchronize: bool = True,
    continuous_measure: bool = True,
) -> float:

    for _ in range(num_warmups):
        _ = model.forward(input_tensor)
    torch.cuda.synchronize()

    elapsed_time_ms = 0

    if continuous_measure:
        start = timer()
        for _ in range(num_repeats):
            _ = model.forward(input_tensor)
        if synchronize:
            torch.cuda.synchronize()
        end = timer()
        elapsed_time_ms = (end - start) * 1000

    else:
        for _ in range(num_repeats):
            start = timer()
            _ = model.forward(input_tensor)
            if synchronize:
                torch.cuda.synchronize()
            end = timer()
            elapsed_time_ms += (end - start) * 1000

    return elapsed_time_ms / num_repeats


@torch.no_grad()
def measure_time_device(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_repeats: int = 100,
    num_warmups: int = 10,
    synchronize: bool = True,
    continuous_measure: bool = True,
) -> float:

    for _ in range(num_warmups):
        _ = model.forward(input_tensor)
    torch.cuda.synchronize()

    elapsed_time_ms = 0

    if continuous_measure:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_repeats):
            _ = model.forward(input_tensor)
        end_event.record()
        if synchronize:
            # This has to be synchronized to compute the elapsed time.
            # Otherwise, there will be runtime error.
            torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

    else:
        for _ in range(num_repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = model.forward(input_tensor)
            end_event.record()
            if synchronize:
                # This has to be synchronized to compute the elapsed time.
                # Otherwise, there will be runtime error.
                torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)

    return elapsed_time_ms / num_repeats


@torch.no_grad()
def run_inference(model: nn.Module,
                  input_tensor: torch.Tensor) -> torch.Tensor:

    return model.forward(input_tensor)


def main() -> None:

    num_warmups = 5
    num_repeats = 10
    input_shape = (4, 1, 16, 64, 64)

    # check if we have a gpu
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device("cuda:0")
    else:
        print("GPU is not available")
        device = torch.device("cpu")

    model = Noise2NoiseUNet3D(in_channels = 1, out_channels = 1, f_maps=16, num_groups=8, number_of_fmaps=5, final_sigmoid = False).to(device)

    model.to(device)
    model.eval()

    # Input tensor
    input_tensor = torch.rand(input_shape, device=device)

    if device == torch.device("cuda:0"):
        torch.cuda.synchronize()

    print("Latency Measurement Using CPU Timer...")
    for continuous_measure in [True, False]:
        for synchronize in [True, False]:
            try:
                latency_ms = measure_time_host(
                    model=model,
                    input_tensor=input_tensor,
                    num_repeats=num_repeats,
                    num_warmups=num_warmups,
                    synchronize=synchronize,
                    continuous_measure=continuous_measure,
                )
                print(f"|"
                      f"Synchronization: {synchronize!s:5}| "
                      f"Continuous Measurement: {continuous_measure!s:5}| "
                      f"Latency: {latency_ms:.5f} ms| ")
            except Exception as e:
                print(f"|"
                      f"Synchronization: {synchronize!s:5}| "
                      f"Continuous Measurement: {continuous_measure!s:5}| "
                      f"Latency: N/A     ms| ")
            if device == torch.device("cuda:0"):
                torch.cuda.synchronize()

    print("Latency Measurement Using CUDA Timer...")
    for continuous_measure in [True, False]:
        for synchronize in [True, False]:
            try:
                latency_ms = measure_time_device(
                    model=model,
                    input_tensor=input_tensor,
                    num_repeats=num_repeats,
                    num_warmups=num_warmups,
                    synchronize=synchronize,
                    continuous_measure=continuous_measure,
                )
                print(f"|"
                      f"Synchronization: {synchronize!s:5}| "
                      f"Continuous Measurement: {continuous_measure!s:5}| "
                      f"Latency: {latency_ms:.5f} ms| ")
            except Exception as e:
                print(f"|"
                      f"Synchronization: {synchronize!s:5}| "
                      f"Continuous Measurement: {continuous_measure!s:5}| "
                      f"Latency: N/A     ms| ")
            if device == torch.device("cuda:0"):
                torch.cuda.synchronize()

    print("Latency Measurement Using PyTorch Benchmark...")
    num_threads = 1
    timer = benchmark.Timer(stmt="run_inference(model, input_tensor)",
                            setup="from __main__ import run_inference",
                            globals={
                                "model": model,
                                "input_tensor": input_tensor
                            },
                            num_threads=num_threads,
                            label="Latency Measurement",
                            sub_label="torch.utils.benchmark.")

    profile_result = timer.timeit(num_repeats)
    # https://pytorch.org/docs/stable/_modules/torch/utils/benchmark/utils/common.html#Measurement
    print(f"Latency: {profile_result.mean * 1000:.5f} ms")


if __name__ == "__main__":

    main()