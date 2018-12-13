import pandas as pd
import numpy as numpy

def EstudiarIDJM(data):
    # data_edad_14_17, data_edad_18_21, data_edad_22_25, data_edad_25_28 = DividirDataPorRangoEdad(data)
    #
    # data_edad_14_17_mujer, data_edad_14_17_hombre = DividirDataPorSexo(data_edad_14_17)
    # data_edad_18_21_mujer, data_edad_18_21_hombre = DividirDataPorSexo(data_edad_18_21)
    # data_edad_22_25_mujer, data_edad_22_25_hombre = DividirDataPorSexo(data_edad_22_25)
    # data_edad_25_28_mujer, data_edad_25_28_hombre = DividirDataPorSexo(data_edad_25_28)

    #fig, (ax1, ax2) = plt.subplots(1, 1, sharey=True, sharex=True)
    #ax1 = data_edad_14_17_mujer['IDJM'].plot.box()
    #ax1.set_title('De 14 a 17')
    #ax2 = data_edad_14_17_hombre['IDJM'].plot.box()
    #ax2.set_title('De 14 a 17: Hombre')

    # plt.boxplot([data_edad_14_17_mujer['IDJM'], data_edad_14_17_hombre['IDJM']],
    #             labels=['Mujer', 'Hombre'])
    # plt.show()

    BoxPlot(data, xname='Sexo',                        yname='IDJM', hue = 'Rango Edad', ymin=30, ymax=110)
    BoxPlot(data, xname='Rango Edad',                  yname='IDJM', hue = 'Sexo',       ymin=30, ymax=110)
    BoxPlot(data, xname='Estrato',                     yname='IDJM', hue = 'Rango Edad', ymin=30, ymax=110)
    BoxPlot(data, xname='Tiempo de Vivir en Medellin', yname='IDJM', hue = 'Rango Edad', ymin=30, ymax=110)
    BoxPlot(data, xname='Comuna',                      yname='IDJM', hue = 'Rango Edad', ymin=30, ymax=110)

    Histogram(data, xname='IDJM', yname='# de Jovenes', nbins=25, grupo=None)
    Histogram(data, xname='IDJM', yname='# de Jovenes', nbins=25, grupo='Sexo')
    Histogram(data, xname='IDJM', yname='# de Jovenes', nbins=25, grupo='Rango Edad')
    Histogram(data, xname='IDJM', yname='# de Jovenes', nbins=25, grupo='Estrato')
    Histogram(data, xname='IDJM', yname='# de Jovenes', nbins=25, grupo='Comuna')



    # ax_estrato = sns.boxplot(x='Estrato', y='IDJM', hue='Rango Edad', data=data)
    # ax_estrato.set_ylim(30,110)
    # plt.savefig('Figures/IDJM_estrato')
    #
    # ax_edad = sns.boxplot(x='Rango Edad', y='IDJM', hue='Sexo', data=data)
    # ax_edad.set_ylim(30,110)
    # plt.savefig('Figures/IDJM_edad')
    #
    #
    #
    # ax_tiempo = sns.boxplot(x='Tiempo de Vivir en Medellin', y='IDJM', hue='Rango Edad', data=data)
    # ax_tiempo.set_ylim(30,110)
    # plt.savefig('Figures/IDJM_tiempo')
def Histogram(data, xname, yname, nbins, grupo):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pylab as pl
    import scipy.stats as stats
    import numpy as np

    title = ''
    if grupo == None:
        title = 'IDJM: ' + 'Total'
        mean   = data[xname].mean()
        median = data[xname].median()
        sigma  = data[xname].std()

        print('mean:', mean)
        print('median:', median)
        print('sigma:', sigma)
    else:
        title = 'IDJM: ' + grupo

    ax = data.hist(column=xname, bins=nbins, by=grupo, sharex=True, figsize=(15,10))


    pl.suptitle(title)
    pl.xlabel(xname)
    pl.ylabel(yname)



    #ax.set_xtitle(xname)
    pl.show()

def BoxPlot(data, xname, yname, hue, ymin, ymax):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.title('Indice de Desarrollo Juvenil de Medellin (2018)')
    ax = sns.boxplot(x=xname, y=yname, hue=hue, data=data)
    ax.set_ylim(ymin,ymax)
    save_name = 'Figures/' + yname + '_' + xname
    plt.savefig(save_name)
    plt.clf()


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
