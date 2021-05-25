import matplotlib.pyplot as plt

if __name__ == '__main__':
    arr = [0,1,2,3,2,1]
    plt.plot(arr,color = 'r')
    plt.title('plt_plot test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()