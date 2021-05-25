import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':


    arr = np.array([3,2,1,1,2,1])
    binX = np.arange(len(arr))
    plt.bar(binX,arr,width=0.8,color = 'g')
    plt.title('plt_bar test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()