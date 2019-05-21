from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#TODO : save plot
def plot_loss(train_loss, validation_loss):
    plt.figure()
    plt.plot(train_loss, c='b', label='Train')
    plt.plot(validation_loss, c='g', label='Valid')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()

#TODO : save plot
#Source : https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7
def plot_confusion_matrix(cm, labels, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      cm :       the confusion matrix
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      figsize:   the size of the figure plotted.
    """

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.show()
    
    return fig
    #plt.savefig(filename)