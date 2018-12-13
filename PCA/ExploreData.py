import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def SinglePlot(data, xTitle, yTitle, Title):
    print(data.value_counts())

    ax = data.value_counts().plot(kind='bar', figsize=(6,6), title=Title)
    #x = data.cla
    #plt.bar(data)
    ax.set_xlabel(xTitle)
    ax.set_ylabel(yTitle)
    #plt.xticks(rotation=90)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    plt.show()

def MultiPlot(data1, data2, xTitle, yTitle):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,6))
    ax1.scatter(data1[0], data1[1], color='red',marker='o', alpha=0.5)
    ax2.scatter(data2[0], data2[1], color='red',marker='o', alpha=0.5)
    Title=['Scaled', 'Not Scaled']

    index = 0
    for ax in (ax1, ax2):
        ax.set_xlabel(xTitle)
        ax.set_ylabel(yTitle)
        ax.set_title(Title[index])
        index += 1

    plt.show()
