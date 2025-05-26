import numpy as np

def gradient_descent(learning_rate=0.1, iterations=100, initial_point=(1.0, 1.0)):

    x0, x1 = initial_point
    history = [(x0, x1)]  
    
    for _ in range(iterations):

        grad_x0 = 2 * x0
        grad_x1 = 2 * x1

        x0 -= learning_rate * grad_x0
        x1 -= learning_rate * grad_x1
        

        history.append((x0, x1))
        
    return x0, x1, history

final_x0, final_x1, path = gradient_descent(learning_rate=0.1, iterations=100, initial_point=(1.0, 1.0))
f_min = final_x0**2 + final_x1**2

print("최종 x0 값:", final_x0)
print("최종 x1 값:", final_x1)
print("f(x0, x1)의 최솟값:", (f_min))
