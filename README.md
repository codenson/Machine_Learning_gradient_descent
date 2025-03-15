# Linear Regression with Gradient Descent

This project implements a simple linear regression model using gradient descent algorithm in Python. The implementation is based on the concepts from a machine learning lab focused on linear regression.

## Overview

The code demonstrates how to:
1. Implement a linear regression model from scratch
2. Use gradient descent to optimize model parameters
3. Visualize the gradient descent process
4. Make predictions using the trained model

## Dependencies

- NumPy
- Matplotlib
- lab_utils_uni (custom utilities for visualization)

## Dataset

The project uses a minimal training dataset with two examples:
- House sizes (in 1000 sqft): [1.0, 2.0]
- House prices (in $1000): [300.0, 500.0]

## Implementation Details

### Cost Function

The cost function calculates the mean squared error between predictions and actual values:

```python
def compute_cost(x, y, w, b):
    m = x.shape[0]  # number of training examples
    total_cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b  # prediction
        cost = (f_wb - y[i])**2  # squared error
        total_cost = 1/(2*m) * cost  # accumulate cost
    return total_cost
```

### Gradient Computation

The `compute_gradient` function calculates the partial derivatives of the cost function with respect to parameters w and b:

```python
def compute_gradient(x, y, w, b):
    m = x.shape[0]  # number of training examples
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = w * x[i] + b  # prediction
        dj_dw += (f_wb - y[i]) * x[i]  # derivative w.r.t. w
        dj_db += (f_wb - y[i])  # derivative w.r.t. b
    dj_dw = (1/m) * dj_dw
    dj_db = (1/m) * dj_db
    
    return dj_dw, dj_db
```

### Gradient Descent

The gradient descent algorithm updates the parameters w and b iteratively to minimize the cost function:

```python
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
    return w, b
```

## Usage

1. Clone the repository
2. Run the script:
   ```
   python linear_regression.py
   ```
3. The script will:
   - Train the model using gradient descent
   - Display visualizations of the gradient descent process
   - Print the final parameters (w and b)
   - Make predictions for houses of different sizes

## Results

After training, the model can predict house prices based on their size. For example:
- A 1000 sqft house: ~$200,000
- A 1200 sqft house: ~$240,000
- A 2000 sqft house: ~$400,000

## Notes

- This implementation is simplified for educational purposes
- The learning rate (alpha) is set to 0.1
- The number of iterations is set to 1000
- The initial parameters are w=2 and b=1.5
