import pandas as pd
import numpy as numpy

def EstudiarIDJM(data):
    # data_edad_14_17, data_edad_18_21, data_edad_22_25, data_edad_25_28 = DividirDataPorRangoEdad(data)
    #
    # data_edad_14_17_mujer, data_edad_14_17_hombre = DividirDataPorSexo(data_edad_14_17)
    # data_edad_18_21_mujer, data_edad_18_21_hombre = DividirDataPorSexo(data_edad_18_21)
    # data_edad_22_25_mujer, data_edad_22_25_hombre = DividirDataPorSexo(data_edad_22_25)
    # data_edad_25_28_mujer, data_edad_25_28_hombre = DividirDataPorSexo(data_edad_25_28)

    #---------------- Categorical Plots ---------------------------
    CatPlot(data, col='Estrato', figsize=(10,6), ymin=0, ymax=80)
    CatPlot(data, col='Comuna', figsize=(10,6), ymin=0, ymax=80)

    #---------------- Bar Plots ---------------------------
    BarPlot(data, xname='Sexo',       yname='IDJM', hue = 'Rango Edad', ymin=0, ymax=80, figsize=(6,6) )
    BarPlot(data, xname='Rango Edad', yname='IDJM', hue = 'Sexo',       ymin=0, ymax=80, figsize=(6,6) )
    BarPlot(data, xname='Estrato',                     yname='IDJM', hue = 'Rango Edad', ymin=0, ymax=80, figsize=(10,6))
    BarPlot(data, xname='Tiempo de Vivir en Medellin', yname='IDJM', hue = 'Rango Edad', ymin=0, ymax=80, figsize=(15,6))
    BarPlot(data, xname='Comuna',                      yname='IDJM', hue = 'Rango Edad', ymin=0, ymax=80, figsize=(21,6))

    #---------------- Box Plots ---------------------------
    BoxPlot(data, xname='Sexo',                        yname='IDJM', hue = 'Rango Edad', ymin=30, ymax=110)
    BoxPlot(data, xname='Rango Edad',                  yname='IDJM', hue = 'Sexo',       ymin=30, ymax=110)
    BoxPlot(data, xname='Estrato',                     yname='IDJM', hue = 'Rango Edad', ymin=30, ymax=110)
    BoxPlot(data, xname='Tiempo de Vivir en Medellin', yname='IDJM', hue = 'Rango Edad', ymin=30, ymax=110)
    BoxPlot(data, xname='Comuna',                      yname='IDJM', hue = 'Rango Edad', ymin=30, ymax=110)

    #---------------- Histogram Plots ---------------------------
    Histogram(data, xname='IDJM', yname='# de Jovenes', nbins=25, grupo=None)
    Histogram(data, xname='IDJM', yname='# de Jovenes', nbins=25, grupo='Sexo')
    Histogram(data, xname='IDJM', yname='# de Jovenes', nbins=25, grupo='Rango Edad')
    Histogram(data, xname='IDJM', yname='# de Jovenes', nbins=25, grupo='Estrato')
    Histogram(data, xname='IDJM', yname='# de Jovenes', nbins=25, grupo='Comuna')

def Histogram(data, xname, yname, nbins, grupo):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pylab as pl
    import scipy.stats as stats
    import numpy as np

    title = ''
    mean = 0.0; sigma = 0.0; y_max = 0.0
    if grupo == None:
        title = 'IDJM ' + 'Total'
        mean   = data[xname].mean()
        median = data[xname].median()
        sigma  = data[xname].std()
        y_max = data[xname].max()

        # print('mean:', mean)
        # print('median:', median)
        # print('sigma:', sigma)
    else:
        title = 'IDJM ' + grupo
        #print( data[xname].groupby(data[grupo]).mean().shape )

    ax = data.hist(column=xname, bins=nbins, by=grupo, sharex=True, figsize=(15,10))
    index = 0
    for plot in ax.flatten():
        plot.set_xlabel(xname)
        plot.set_ylabel(yname)
        plot.grid(axis='y', alpha=0.75)

        if grupo == None:
            plot.text(mean,    plot.get_ylim()[1], 'media = %0.1f' %mean)
            plot.text(mean+15, plot.get_ylim()[1], 'std = %0.1f' %sigma)
        else:
            #print('grupo:', grupo, ' index:', index)
            if grupo == 'Estrato':
                mean  = data[xname].groupby(data[grupo]).mean()[index+1]
                sigma = data[xname].groupby(data[grupo]).std()[index+1]
            else:
                #print('grupo:', grupo, ' index:', index)
                if index == 21: continue

                mean  = data[xname].groupby(data[grupo]).mean()[index]
                sigma = data[xname].groupby(data[grupo]).std()[index]

            plot.text(0,       plot.get_ylim()[1], 'media = %0.1f' %mean)
            plot.text(mean+15, plot.get_ylim()[1], 'std = %0.1f' %sigma)
            index += 1


    pl.suptitle(title)
    pl.xlabel(xname)
    pl.ylabel(yname)
    pl.grid(axis='y', alpha=0.75)

    save_name = 'Figures/' + 'Histogram' + '_' + title
    pl.savefig(save_name)

def BoxPlot(data, xname, yname, hue, ymin, ymax):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.title('Indice Promedio de Desarrollo Juvenil de Medellin (2018)')
    ax = sns.boxplot(x=xname, y=yname, hue=hue, data=data)
    ax.set_ylim(ymin,ymax)
    save_name = 'Figures/' + yname + '_' + xname
    plt.savefig(save_name)
    plt.clf()

def BarPlot(data, xname, yname, hue, ymin, ymax, figsize):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.title('Indice Promedio de Desarrollo Juvenil de Medellin (2018)')
    ax = sns.barplot(x=xname, y=yname, hue=hue, data=data, ci="sd", capsize=.1)
    ax.set_ylim(ymin,ymax)

    save_name = 'Figures/BarPlot' + yname + '_' + xname
    plt.savefig(save_name)
    plt.clf()

def CatPlot(data, col, figsize, ymin, ymax):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.title('Indice Promedio de Desarrollo Juvenil de Medellin (2018)')
    ax = sns.catplot(x='Sexo', y = 'IDJM', hue='Rango Edad',
                     col=col, data=data, kind='bar', col_wrap=3, ci='sd',
                     capsize=.1, height=2.5, aspect=.8)
    #ax.set_ylim(ymin,ymax)

    plt.show()

def DividirDataPorSexo(data):
    mujer  = GetData(data, col='Sexo', item='Mujer')
    hombre = GetData(data, col='Sexo', item='Hombre')

    return mujer, hombre

def DividirDataPorRangoEdad(data):

    edad_14_17 = GetData(data, col='Rango Edad', item='De 14 a 17 años')
    edad_18_21 = GetData(data, col='Rango Edad', item='De 18 a 21 años')
    edad_22_25 = GetData(data, col='Rango Edad', item='De 22 a 25 años')
    edad_25_28 = GetData(data, col='Rango Edad', item='De 25 a 28 años')

    return edad_14_17, edad_18_21, edad_22_25, edad_25_28

def GetData(data, col, item):
    esItem = data[col]==item
    data_item = data[esItem]

    return data_item
