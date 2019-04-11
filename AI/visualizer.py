from matplotlib import pyplot as plt

def plot_loss(train_loss, validation_loss):
    plt.figure()
    plt.plot(train_loss, c='b', label='Train')
    plt.plot(validation_loss, c='g', label='Valid')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()