# Cpp-AI-Algorithms

A collection of AI/ML algorithms implemented in C++ for performance-critical applications.

## Overview

This repository hosts a growing collection of fundamental and advanced Artificial Intelligence and Machine Learning algorithms, meticulously implemented in C++. The primary goal is to provide highly optimized, low-latency solutions for scenarios where computational efficiency is paramount, such as embedded systems, real-time processing, and high-performance computing environments.

## Features

*   **Core ML Algorithms:** Implementations of algorithms like K-Means, SVM, Decision Trees, and basic Neural Networks.
*   **Optimized Data Structures:** Custom data structures designed for efficient numerical operations.
*   **Performance Benchmarking:** Tools and examples for benchmarking algorithm performance.
*   **Integration Examples:** Demonstrations of how to integrate these C++ algorithms into larger systems.

## Getting Started

### Prerequisites

*   C++ Compiler (GCC/Clang)
*   CMake

### Building the Algorithms

```bash
git clone https://github.com/Pects1949/Cpp-AI-Algorithms.git
cd Cpp-AI-Algorithms
mkdir build
cd build
cmake ..
make
```

## Usage (Example - K-Means Clustering)

```cpp
// src/kmeans.cpp
#include <iostream>
#include <vector>
#include <random>
#include <limits>

// Simple K-Means implementation (for demonstration purposes)
namespace AIAlgorithms {

struct Point {
    double x, y;
};

double euclideanDistance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

std::vector<Point> runKMeans(const std::vector<Point>& data, int k, int max_iterations) {
    std::vector<Point> centroids(k);
    std::vector<int> assignments(data.size());

    // 1. Initialize centroids randomly
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, data.size() - 1);
    for (int i = 0; i < k; ++i) {
        centroids[i] = data[distrib(gen)];
    }

    for (int iter = 0; iter < max_iterations; ++iter) {
        // 2. Assign points to nearest centroid
        for (size_t i = 0; i < data.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int closest_centroid = -1;
            for (int j = 0; j < k; ++j) {
                double dist = euclideanDistance(data[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }
            assignments[i] = closest_centroid;
        }

        // 3. Update centroids
        std::vector<Point> new_centroids(k, {0.0, 0.0});
        std::vector<int> counts(k, 0);
        for (size_t i = 0; i < data.size(); ++i) {
            new_centroids[assignments[i]].x += data[i].x;
            new_centroids[assignments[i]].y += data[i].y;
            counts[assignments[i]]++;
        }

        for (int i = 0; i < k; ++i) {
            if (counts[i] > 0) {
                centroids[i].x = new_centroids[i].x / counts[i];
                centroids[i].y = new_centroids[i].y / counts[i];
            }
        }
    }
    return centroids;
}

} // namespace AIAlgorithms

int main() {
    std::cout << "C++ AI Algorithms - K-Means Example" << std::endl;
    std::vector<AIAlgorithms::Point> data = {
        {1.0, 1.0}, {1.5, 2.0}, {3.0, 4.0}, {5.0, 7.0},
        {3.5, 5.0}, {4.5, 5.0}, {3.5, 4.5}, {8.0, 8.0}
    };

    int k = 2;
    int max_iterations = 100;

    std::vector<AIAlgorithms::Point> final_centroids = AIAlgorithms::runKMeans(data, k, max_iterations);

    std::cout << "Final Centroids:" << std::endl;
    for (const auto& c : final_centroids) {
        std::cout << "(" << c.x << ", " << c.y << ")" << std::endl;
    }

    return 0;
}

```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for more details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
