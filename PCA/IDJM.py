import pandas as pd
import numpy as np

from ExploreData import SinglePlot, MultiPlot
from EstudiarIDJM import EstudiarIDJM


def main():
    excel_file = 'DataBase/Base_Datos_Medellin_2018.xlsx'
    data_table = pd.read_excel(excel_file)

    primera_componente_trabajo,     variables_trabajo     = CalcularIDJM_Trabajo(data_table)
    primera_componente_educacion,   variables_educacion   = CalcularIDJM_Educacion(data_table)
    primera_componente_familia,     variables_familia     = CalcularIDJM_Familia(data_table)
    primera_componente_bienes,      variables_bienes      = CalcularIDJM_Bienes(data_table)
    primera_componente_democracia,  variables_democracia  = CalcularIDJM_Democracia(data_table)
    primera_componente_salud,       variables_salud       = CalcularIDJM_Salud(data_table)
    primera_componente_convivencia, variables_convivencia = CalcularIDJM_Convivencia(data_table)
    primera_componente_desarrollo,  variables_desarrollo  = CalcularIDJM_Desarrollo(data_table)

    Trabajo     = np.dot(variables_trabajo,     primera_componente_trabajo)
    Educacion   = np.dot(variables_educacion,   primera_componente_educacion)
    Familia     = np.dot(variables_familia,     primera_componente_familia)
    Bienes      = np.dot(variables_bienes,      primera_componente_bienes)
    Democracia  = np.dot(variables_democracia,  primera_componente_democracia)
    Salud       = np.dot(variables_salud,       primera_componente_salud)
    Convivencia = np.dot(variables_convivencia, primera_componente_convivencia)
    Desarrollo  = np.dot(variables_desarrollo,  primera_componente_desarrollo)

    data = pd.DataFrame( {'Educacion':Educacion,
                          'Trabajo':Trabajo,
                          'Familia':Familia,
                          'Bienes y Servicios':Bienes,
                          'Democracia y Participacion':Democracia,
                          'Salud':Salud,
                          'Convivencia':Convivencia,
                          'Desarrollo':Desarrollo} )
    print(data.head())

    IDJM_primera_componente, IDJM_variables = ProcessPCA(data)

    puntajes = np.dot(IDJM_variables, IDJM_primera_componente)
    IDJM_puntajes = GetIDJM(puntajes)
    print('IDJM Total: ',      GetIDJM(puntajes) )

    data_table['Sexo'].replace('Otro','Mujer',inplace=True)
    print(data_table['Sexo'].value_counts())
    data_table['comuna'].replace('castilla', 'Castilla', inplace=True)

    data_info = pd.DataFrame({'Comuna':data_table['comuna'],
                              'Sexo':data_table['Sexo'],
                              'Estrato':data_table['estrato'],
                              'Rango Edad':data_table['Rango edad'],
                              'Tiempo de Vivir en Medellin':data_table['tiem_vivir_med'],
                              'IDJM':IDJM_puntajes})

    print(data_info.head())

    finalDf = pd.concat([data_info, data], axis=1)
    print(finalDf.head())

    EstudiarIDJM(finalDf)
    # print('IDJM Trabajo: ',    GetIDJM(Trabajo) )
    # print('IDJM Educacion: ',  GetIDJM(Educacion) )
    # print('IDJM Familia: ',    GetIDJM(Familia) )
    # print('IDJM Bienes: ',     GetIDJM(Bienes) )
    # print('IDJM Democracia: ', GetIDJM(Democracia) )

    #IDJM_data = pd.DataFrame()

    # print('IDJM Trabajo:',   GetIDJM(primera_componente_trabajo))
    # print('IDJM Educacion:', GetIDJM(primera_componente_educacion))
    # print('IDJM Familia:',   GetIDJM(primera_componente_familia))
    #print(comp_principal)

def GetIDJM(primera_componente):
    #print(type(primera_componente))
    min_index = np.argmin(primera_componente)
    max_index = np.argmax(primera_componente)

    minZ = primera_componente[min_index]
    maxZ = primera_componente[max_index]

    sigma = 100.0/(maxZ - minZ)
    mu = -(sigma*minZ)

    X = sigma*primera_componente + mu

    #print(Z)
    return X

def CalcularIDJM_Salud(data_table):
    feature_names = ['33a','33b','33c',
                     '35a','35b','35c','35d','35e',
                      37,
                     '38a','38b','38c','38d',
                     '39a','39b',
                     '41a','41b','41c','41d',
                      42,
                     '44a','44b','44c','44d',
                      45,
                     '47a','47b','47c','47d',
                      48,
                     '49a','49b','49c','49d','49e','49f',
                     '50a','50b',
                      55,
                     '64hs']
    X = data_table[feature_names]

    X['33a'].fillna(1, inplace=True)
    X['33b'].fillna(1, inplace=True)
    X['33c'].fillna(1, inplace=True)

    salud = CodificarEtiqueta(X, feature_names)
    pc_salud, var_trans_salud = ProcessPCA(salud)
    return pc_salud, var_trans_salud

def CalcularIDJM_Desarrollo(data_table):
    feature_names = ['56a','56b','56c','56d','56e']
    X = data_table[feature_names]

    proyecto_vida = CalcularCalidad(data_table, features=['57a','57b','57c','57d','57e','57f','57g','57h'])
    respeto = CalcularCalidad(data_table, features=['64as','64bs','64cs','64ds','64es','64fs','64gs','64hs','64is','64js','64ms','64ls'])
    #SinglePlot(respeto, xTitle='', yTitle='Frecuencia', Title='Te Respetan?')

    proyecto_vida.rename('Satisfaccion', inplace=True)
    respeto.rename('Respeto Entorno', inplace=True)

    Desarrollo = pd.concat([X['56a'],X['56b'],X['56c'],X['56d'],X['56e'], proyecto_vida, respeto], axis=1)
    new_features = ['56a','56b','56c','56d','56e', 'Satisfaccion', 'Respeto Entorno']

    desarrollo = CodificarEtiqueta(Desarrollo, new_features)
    pc_desarrollo, var_trans_desarrollo = ProcessPCA(desarrollo)
    return pc_desarrollo, var_trans_desarrollo

def CalcularIDJM_Convivencia(data_table):
    feature_names = ['51a','51b','51c',
                     '52a','52b',
                      53,
                     '64as','64bs','64cs','64ds','64es','64fs']

    X = data_table[feature_names]

    X['51a'].fillna(1, inplace=True)
    X['51b'].fillna(1, inplace=True)
    X['51c'].fillna(1, inplace=True)
    # SinglePlot(X['51a'], xTitle='Amenazas',         yTitle='Frecuencia', Title='Grupos Armados')
    # SinglePlot(X['51b'], xTitle='Aciones Forzadas', yTitle='Frecuencia', Title='Grupos Armados')
    # SinglePlot(X['51c'], xTitle='Amenazas',         yTitle='Frecuencia', Title='Personas Entorno')
    #amenazas = CalcularCalidad(X, features=['51a','51b','51c'])

    #SinglePlot(X['52a'], xTitle='Fronteras Invisibles', yTitle='Frecuencia', Title='Grupos Armados')
    #SinglePlot(X['52b'], xTitle='Seguridad Barrio',     yTitle='Frecuencia', Title='Grupos Armados')
    #seguridad = CalcularCalidad(X, features=['52a','52b'])
    agresiones = CalcularCalidad(data_table, features=['58a','58b','58c','58d','58e','58f','58g','58h'])
    agresiones.rename('Agresiones', inplace=True)

    X[53] = FillSI_NO(X[53])
    X[53].fillna('NO', inplace=True)
    Convivencia = pd.concat([X['51a'],X['51b'],X['51c'],X['52a'],X['52b'],X[53],
                             X['64as'],X['64bs'],X['64cs'],X['64ds'],X['64es'],X['64fs'], agresiones], axis=1)
    new_features = ['51a','51b','51c',
                     '52a','52b',
                      53,
                     '64as','64bs','64cs','64ds','64es','64fs',
                     'Agresiones']

    convivencia = CodificarEtiqueta(Convivencia, new_features)
    pc_convivencia, var_trans_convivencia = ProcessPCA(convivencia)
    return pc_convivencia, var_trans_convivencia

def CalcularIDJM_Democracia(data_table):
    feature_names = ['31a','31b','31c',                    #liderazgo e iniciativas
                     '32a','32b','32c','32d','32e']        #Participacion
    X = data_table[feature_names]

    # Calidad_Liderazgo     = CalcularCalidad(X, features=['31a','31b','31c'])
    # Calidad_Participacion = CalcularCalidad(X, features=['32a','32b','32c','32d','32e'])
    #
    # Democracia = pd.DataFrame({'Liderazgo':Calidad_Liderazgo,
    #                            'Participacion':Calidad_Participacion})
    # new_features = ['Liderazgo', 'Participacion']

    democracia_participacion = CodificarEtiqueta(X, feature_names)
    pc_democracia, var_trans_democracia = ProcessPCA(democracia_participacion)
    return pc_democracia, var_trans_democracia

def CalcularIDJM_Bienes(data_table):
    feature_names = ['27a','27b','27c','27d',                                    #Calidad vivienda
                     '28a','28b','28c','28d','28e','28f','28g',                  #Calidad Servicios Publicos
                     '29a','29b','29c','29d',                                    #Ofertas Ciudad
                     '36a','36b',
                     '49e','49f',
                     '64gs']
    X = data_table[feature_names]

    # Calidad_Vivienda  = CalcularCalidad(X, features = ['27a','27b','27c','27d'])
    # Calidad_Servicios = CalcularCalidad(X, features = ['28a','28b','28c','28d','28e','28f','28g'])
    # Calidad_Ofertas   = CalcularCalidad(X, features = ['29a','29b','29c','29d'])
    #
    # Bienes = pd.DataFrame({'Tipo de Vivienda':X[26],
    #                        'Calidad de Vivienda':Calidad_Vivienda,
    #                        'Calidad de Servicios Publicos':Calidad_Servicios,
    #                        'Ofertas de la Ciudad':Calidad_Ofertas,
    #                        'Gastos Semanales':X['30hh']})
    #
    # #print(Bienes.head())
    # new_features = ['Tipo de Vivienda', 'Calidad de Vivienda', 'Calidad de Servicios Publicos',
    #                 'Ofertas de la Ciudad', 'Gastos Semanales']

    bienes_servicios = CodificarEtiqueta(X, feature_names)
    pc_bienes, var_trans_bienes = ProcessPCA(bienes_servicios)

    return pc_bienes, var_trans_bienes

def CalcularIDJM_Familia(data_table):
    feature_names = [23,
                    '25a','25b','25c','25d',
                    '59a',
                    '64af','64bf','64cf','64df','64ef','64ff','64gf','64hf','64if','64jf','64mf','64lf']  # pregunta 21 parece ser informacion irrelevante, es resumida en la pregunta 22 (numero de personas que habitan la casa)
                                                                             # preguntas 24 c a f dependen de b, irrelevante en este estudio
    X = data_table[feature_names]
    #X['24a'].fillna('No value', inplace=True)
    #X['24b'].fillna('No value', inplace=True)

    respeto = CalcularCalidad(X, features=['64af','64bf','64cf','64ef','64ff','64gf','64hf','64if','64jf','64mf','64lf'])
    #SinglePlot(respeto, xTitle='', yTitle='Frecuencia', Title='Tu familia te Respeta?')
    respeto.rename('Respeto Familia', inplace=True)

    print(respeto.head())
    data = pd.concat([X[23], X['25a'],X['25b'],X['25c'],X['25d'],X['59a'],X['64df'], respeto], axis=1)
    print(data.head())

    new_features = [23,
                    '25a','25b','25c','25d',
                    '59a',
                    '64df',
                    'Respeto Familia']

    familia = CodificarEtiqueta(data, new_features)
    pc_familia, var_trans_familia = ProcessPCA(familia)

    return pc_familia, var_trans_familia

def CalcularIDJM_Educacion(data_table):
    feature_names = ['pre_11',
                     '12a','12b','12c','12d','12e','12f','12g','12h','12i',
                     '13a','13b','13c','13d',
                     '14a','14b','14c','14d','14e','14f',
                     '59c',
                     '64ae','64be','64ce','64de','64ee','64fe','64ge','64he','64ie','64je','64me','64le']

    #X = data_table[feature_names]
    #max_grado_escolar = FindMaxEscolar(X)
    idioma = FindIdioma(data_table)

    data_table['pre_11'] = FillSI_NO(data_table['pre_11'])
    #SinglePlot(X['pre_11'], ' ', '# de Jovenes', 'Estudia Actualmente?')


    data = pd.concat([idioma, data_table[feature_names]], axis=1)

    educacion    = CodificarEtiqueta(data, feature_names)
    #MultiPlot(educacion)
    pc_educacion, var_trans_educacion = ProcessPCA(educacion)

    return pc_educacion, var_trans_educacion

def FindMaxEscolar(X):
    feature_names = ['9a', '9b', '9c', '9d', '9e', '9f']

    data_edu    = X[feature_names]
    universidad = data_edu['9e']
    tecnologia  = data_edu['9d']

    tecnica     = data_edu['9c']
    colegio     = data_edu['9b']
    primaria    = data_edu['9a']

    universidad.replace('Completa',              'Uni_comp',   inplace=True)
    universidad.replace('Incompleta o en curso', 'Uni_incomp', inplace=True)
    universidad.replace('Incompleta o en curs',  'Uni_incomp', inplace=True)

    tecnologia.replace('Completa',              'tecno_comp',   inplace=True)
    tecnologia.replace('Incompleta o en curso', 'tecno_incomp', inplace=True)
    tecnologia.replace('Incompleta o en curs',  'tecno_incomp', inplace=True)

    tecnica.replace('Completa',              'tecnica_comp',   inplace=True)
    tecnica.replace('Incompleta o en curso', 'tecnica_incomp', inplace=True)
    tecnica.replace('Incompleta o en curs',  'tecnica_incomp', inplace=True)

    colegio.fillna(primaria, inplace=True)

    universidad.fillna(tecnologia, inplace=True)
    universidad.fillna(tecnica,    inplace=True)
    universidad.fillna(colegio,    inplace=True)

    #SinglePlot(universidad, 'Nivel Educativo', '# de Jovenes', 'Nivel Educativo Jovenes de Medellin')

    return universidad

def CalcularCalidad(X, features):
    data_vivienda = X[features]
    data_median = data_vivienda.median(1)
    data_median.fillna(0, inplace=True)
    data_median = data_median.astype(int)
    #print(data_median.head())
    return data_median

def FindIdioma(X):
    features = ['10a','10b','10c','10d','10e','10f','10g','10h']
    data_idioma = X[features]

    idioma = data_idioma.count(axis=1)
    print('Idiomas: ', idioma.head())
    #SinglePlot(idioma, xTitle='Idiomas', yTitle='Frecuencia', Title='Habilidad Trabajo')
    return idioma

def CalcularIDJM_Trabajo(data_table):

    #feature_names = [16, 17, '18a', '18b', '18c','18d','19a','19b']  # preguntas 17 y 18 son codicionales a 17, no usar en el analisis
    feature_names = [16,'18a','18b','18c','18d','19a','19b','59b']
    X = data_table[feature_names]

    # SinglePlot(X['18a'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')
    # SinglePlot(X['18b'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')
    # SinglePlot(X['18c'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')
    # SinglePlot(X['18d'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')
    # SinglePlot(X['19a'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')
    # SinglePlot(X['19b'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')
    # SinglePlot(X['59b'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')

    X['19a'].fillna(0, inplace=True)
    X['19b'].fillna(0, inplace=True)

    trabajo = CodificarEtiqueta(X, feature_names)
    pc_trabajo, var_trans_trabajo = ProcessPCA(trabajo)

    return pc_trabajo, var_trans_trabajo

def ProcessPCA(dataset):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    sc = StandardScaler()
    std_dataset = sc.fit_transform(dataset)

    pca = PCA()
    pca_dataset = pca.fit_transform(std_dataset)
    #NoScale_pca_dataset = pca.fit_transform(dataset)
    #MultiPlot(pca_dataset, NoScale_pca_dataset, xTitle='1era Componente', yTitle='2da Componente')
    #singular_values = pca.singular_values_
    primera_comp = pca.components_[0]
    print('Forma: ' , pca_dataset.shape)
    print('Primera Comp: ' , pca.components_[0])
    print('Ratio variance: ', pca.explained_variance_ratio_)
    return primera_comp, pca_dataset

def CodificarEtiqueta(array, column):
    import category_encoders as ce

    encoder = ce.backward_difference.BackwardDifferenceEncoder(cols=column)
    encoder.fit(array, verbose=1)

    ret_df = encoder.transform(array)
    return ret_df


def FillSI_NO(array):
    array.fillna('NO', inplace=True)
    array.replace('si', 'SI', inplace=True)
    array.replace('Si', 'SI', inplace=True)
    array.replace('sI', 'SI', inplace=True)
    array.replace('no', 'NO', inplace=True)
    array.replace('No', 'NO', inplace=True)
    array.replace('nO', 'NO', inplace=True)
    return array


if __name__ == '__main__':
  main()
