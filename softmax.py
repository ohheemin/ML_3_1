import matplotlib.pyplot as plt
import numpy as np

# MNIST 데이터 로드
(train_img, train_label), (test_img, test_label) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

def display_mnist_images(images, labels, num_images=10):
    """MNIST 이미지 출력 함수"""
    plt.figure(figsize=(10, 2))  # 그래프 크기 설정
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"{labels[i]}")
        plt.axis('off')
    plt.show()

# 학습 데이터 중 10개 출력
display_mnist_images(train_img, train_label, num_images=10)