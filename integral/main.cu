#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "cfg_reader.h"
#include "functions.cuh"
#include <cuda_runtime.h>

double integrate_function(int function_id, const double x1_start, const double x1_end, const double x2_start, const double x2_end, const int x1_steps, const int x2_steps) {
    double *d_result, h_result = 0.0;
    cudaMalloc(&d_result, sizeof(double));
    cudaMemcpyAsync(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice, 0);

    double dx1 = (x1_end - x1_start) / x1_steps;
    double dx2 = (x2_end - x2_start) / x2_steps;

    dim3 blockSize(16, 16);
    dim3 gridSize((x1_steps + blockSize.x - 1) / blockSize.x, 
                  (x2_steps + blockSize.y - 1) / blockSize.y);
    gridSize.x = min(gridSize.x, 65535);
    gridSize.y = min(gridSize.y, 65535);

    integrate<<<gridSize, blockSize>>>(function_id, x1_start, dx1, x2_start, dx2, x1_steps, x2_steps, d_result);
    
    cudaMemcpyAsync(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost, 0);
    cudaFree(d_result);
    
    cudaDeviceSynchronize();
    return h_result;
}

bool nearlyEqual(const double a, const double b, const double epsilon) {
    const double absA = std::abs(a);
    const double absB = std::abs(b);
    const double diff = std::abs(a - b);
    if (a == b) {
        return true;
    } else if (a == 0 || b == 0 || (absA + absB < std::numeric_limits<double>::min())) {
        return diff < (epsilon * std::numeric_limits<double>::min());
    } else {
        const double d = std::min(absA + absB, std::numeric_limits<double>::max());
        return diff / d < epsilon;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Wrong amount of arguments" << std::endl;
        return 1;
    }

    const std::string function_n = argv[1];

    const std::string file_name = argv[2];
    std::vector<std::string> data = read_file(file_name);
    std::map<std::string, float> config = input_processing(data);

    const double x1_start = config["x_start"];
    const double x1_end = config["x_end"];
    const double x2_start = config["y_start"];
    const double x2_end = config["y_end"];
    double x1_step = config["init_steps_x"];
    double x2_step = config["init_steps_y"];
    const double max_iter = config["max_iter"];
    const double abs_err = config["abs_err"];
    const double rel_err = config["rel_err"];

    double prev_res = 0.0;
    double absolute_error = 0.0;
    double relative_error = 0.0;
    double result = 0.0;

    int selected_function = std::stoi(function_n);
    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < max_iter; i++) {
        result = integrate_function(selected_function, x1_start, x1_end, x2_start, x2_end, x1_step, x2_step);

        if (i != 0 && nearlyEqual(result, prev_res, abs_err) &&
            nearlyEqual(1.0, prev_res / result, rel_err)) {
            absolute_error = std::abs(result - prev_res);
            relative_error = std::abs(result - prev_res) / std::abs(result);
            break;
        }

        prev_res = result;
        x1_step *= 2.0;
        x2_step *= 2.0;
    }
    std::cout << std::fixed << std::setprecision(2) << result << std::endl;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << absolute_error << std::endl;
    std::cout << relative_error << std::endl;

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    return 0;
}