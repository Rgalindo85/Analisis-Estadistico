import pandas as pd
import numpy as np
import math

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

    # print(primera_componente_trabajo.shape)
    #print(variables_educacion)
    # print(primera_componente_familia.shape)
    # print(primera_componente_bienes.shape)
    # print(primera_componente_democracia.shape)
    # print(primera_componente_salud.shape)
    # print(primera_componente_convivencia.shape)
    # print(primera_componente_desarrollo.shape)
    #
    # Trabajo     = variables_trabajo[:,0]
    # Educacion   = variables_educacion[:,0]
    # Familia     = variables_familia[:,0]
    # Bienes      = variables_bienes[:,0]
    # Democracia  = variables_democracia[:,0]
    # Salud       = variables_salud[:,0]
    # Convivencia = variables_convivencia[:,0]
    # Desarrollo  = variables_desarrollo[:,0]


    # Trabajo     = variables_trabajo.T[0]
    # Educacion   = variables_educacion.T[0]
    # Familia     = variables_familia.T[0]
    # Bienes      = variables_bienes.T[0]
    # Democracia  = variables_democracia.T[0]
    # Salud       = variables_salud.T[0]
    # Convivencia = variables_convivencia.T[0]
    # Desarrollo  = variables_desarrollo.T[0]

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
    max_total =  np.amax(IDJM_variables[0]) + np.amax(Trabajo[1]) + np.amax(Familia[2]) + np.amax(Bienes[3]) + np.amax(Democracia[4]) + np.amax(Salud[5]) + np.amax(Convivencia[6]) + np.amax(Desarrollo[7])

    print('max total:', max_total)

    puntajes = np.dot(IDJM_variables, IDJM_primera_componente)
    puntajes *= 1/max_total
    #puntajes = IDJM_variables[:,0]
    #puntajes = IDJM_primera_componente
    #print('Z = ', np.dot(IDJM_variables, IDJM_primera_componente))

    IDJM_puntajes = GetIDJM(puntajes)
    #print('IDJM Total: ',       GetIDJM(puntajes) )
    # print('IDJM Educacion: ',   GetIDJM(Educacion) )
    # print('IDJM Trabajo: ',     GetIDJM(Trabajo) )
    # print('IDJM Familia: ',     GetIDJM(Familia) )
    # print('IDJM Bienes: ',      GetIDJM(Bienes) )
    # print('IDJM Democracia: ',  GetIDJM(Democracia) )
    # print('IDJM Salud: ',       GetIDJM(Salud) )
    # print('IDJM Convivencia: ', GetIDJM(Convivencia) )
    # print('IDJM Desarrollo: ',  GetIDJM(Desarrollo) )


    data_table['Sexo'].replace('Otro','Mujer',inplace=True)
    #print(data_table['Sexo'].value_counts())
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


def GetIDJM(primera_componente):
    print(primera_componente.shape)
    #min_index = np.argmin(primera_componente)
    #max_index = np.argmax(primera_componente)

    minZ = np.amin(primera_componente)
    maxZ = np.amax(primera_componente)

    # mu = primera_componente.mean()
    # sigma = primera_componente.std()
    # primera_componente += abs(minZ)
    # minZ = np.amin(primera_componente)
    # maxZ = np.amax(primera_componente)

    sigma = 100.0/(maxZ - minZ)
    mu =  -(sigma*minZ)


    X = sigma*primera_componente + mu
    print (minZ, maxZ, mu, sigma)

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

    X['33a'] = NumberToAcuerdo_Desarrollo(X['33a'])
    X['33b'] = NumberToAcuerdo_Desarrollo(X['33b'])
    X['33c'] = NumberToAcuerdo_Desarrollo(X['33c'])

    X['35a'] = NumberToAcuerdo_Desarrollo(X['35a'])
    X['35b'] = NumberToAcuerdo_Desarrollo(X['35b'])
    X['35c'] = NumberToAcuerdo_Desarrollo(X['35c'])
    X['35d'] = NumberToAcuerdo_Desarrollo(X['35d'])
    X['35e'] = NumberToAcuerdo_Desarrollo(X['35e'])

    X[37].apply(lambda x: X[37].replace(x, 'SI', inplace=True) if x != 'Ninguno' else x)
    X[37].replace('Ninguno', 'NO', inplace=True)

    X['38a'] = NumberToAcuerdo_Desarrollo(X['38a'])
    X['38b'] = NumberToAcuerdo_Desarrollo(X['38b'])
    X['38c'] = NumberToAcuerdo_Desarrollo(X['38c'])
    X['38d'] = NumberToAcuerdo_Desarrollo(X['38d'])

    X['39a'] = NumberToTiemp_Bienes(X['39a'])
    X['39b'] = NumberToTiemp_Bienes(X['39b'])

    X['41a'] = NumberToTiemp_Bienes(X['41a'])
    X['41b'] = NumberToTiemp_Bienes(X['41b'])
    X['41c'] = NumberToTiemp_Bienes(X['41c'])
    X['41d'] = NumberToTiemp_Bienes(X['41d'])

    X[42] = NumberToAcuerdo_Desarrollo(X[42])

    X['44a'] = NumberToTiemp_Bienes(X['44a'])
    X['44b'] = NumberToTiemp_Bienes(X['44b'])
    X['44c'] = NumberToTiemp_Bienes(X['44c'])
    X['44d'] = NumberToTiemp_Bienes(X['44d'])

    X[45] = NumberToAcuerdo_Desarrollo(X[45])

    X['47a'] = NumberToTiemp_Bienes(X['47a'])
    X['47b'] = NumberToTiemp_Bienes(X['47b'])
    X['47c'] = NumberToTiemp_Bienes(X['47c'])
    X['47d'] = NumberToTiemp_Bienes(X['47d'])

    X[48] = NumberToAcuerdo_Desarrollo(X[48])

    X['49a'] = NumberToTiemp_Bienes(X['49a'])
    X['49b'] = NumberToTiemp_Bienes(X['49b'])
    X['49c'] = NumberToTiemp_Bienes(X['49c'])
    X['49d'] = NumberToTiemp_Bienes(X['49d'])

    X['50a'] = NumberToTiemp_Bienes(X['50a'])
    X['50b'] = NumberToTiemp_Bienes(X['50b'])

    X[55] = NumberToAcuerdo_Desarrollo(X[55])

    X['64hs'] = NumberToTiemp_Bienes(X['64hs'])
    #SinglePlot(X['50a'], xTitle='', yTitle='Frecuencia', Title='Te Respetan?')
    salud = CodificarEtiqueta(X, feature_names)
    pc_salud, var_trans_salud = ProcessPCA(salud)
    return pc_salud, var_trans_salud

def CalcularIDJM_Desarrollo(data_table):
    feature_names = ['56a','56b','56c','56d','56e', '57a']
    X = data_table[feature_names]

    X['56a'] = NumberToCalidad_Bienes(X['56a'])
    X['56b'] = NumberToCalidad_Bienes(X['56b'])
    X['56c'] = NumberToCalidad_Bienes(X['56c'])
    X['56d'] = NumberToCalidad_Bienes(X['56d'])
    X['56e'] = NumberToCalidad_Bienes(X['56e'])

    X['57a'] = NumberToAcuerdo_Desarrollo(X['57a'])

    #proyecto_vida = CalcularCalidad(data_table, features=['57a','57b','57c','57d','57e','57f','57g','57h'])
    respeto = CalcularCalidad(data_table, features=['64as','64bs','64cs','64ds','64es','64fs','64gs','64hs','64is','64js','64ms','64ls'])
    respeto = NumberToCalidad_Bienes(respeto)
    #SinglePlot(respeto, xTitle='', yTitle='Frecuencia', Title='Te Respetan?')

    #proyecto_vida.rename('Satisfaccion', inplace=True)
    respeto.rename('Respeto Entorno', inplace=True)

    Desarrollo = pd.concat([X['56a'],X['56b'],X['56c'],X['56d'],X['56e'], X['57a'], respeto], axis=1)
    new_features = ['56a','56b','56c','56d','56e', '57a', 'Respeto Entorno']

    desarrollo = CodificarEtiqueta(Desarrollo, new_features)
    pc_desarrollo, var_trans_desarrollo = ProcessPCA(desarrollo)
    return pc_desarrollo, var_trans_desarrollo

def NumberToAcuerdo_Desarrollo(df):
    df.replace(1, 'Muy Mala',         inplace=True)
    df.replace(2, 'Mala',             inplace=True)
    df.replace(3, 'Ni Buena Ni Mala', inplace=True)
    df.replace(4, 'Buena',            inplace=True)
    df.replace(5, 'Muy Buena',        inplace=True)
    df.replace(0, 'Otro',             inplace=True)

    return df

def CalcularIDJM_Convivencia(data_table):
    feature_names = ['51a','51b','51c',
                     '52a','52b',
                      53,
                     '64as','64bs','64cs','64ds','64es','64fs']

    X = data_table[feature_names]

    X['51a'].fillna(1, inplace=True)
    X['51b'].fillna(1, inplace=True)
    X['51c'].fillna(1, inplace=True)

    X['51a'] = NumberToVeces_Convivencia(X['51a'])
    X['51b'] = NumberToVeces_Convivencia(X['51b'])
    X['51c'] = NumberToVeces_Convivencia(X['51c'])

    X['52a'] = NumberToAcuerdo_Desarrollo(X['52a'])
    X['52b'] = NumberToAcuerdo_Desarrollo(X['52b'])
    # SinglePlot(X['51a'], xTitle='Amenazas',         yTitle='Frecuencia', Title='Grupos Armados')

    agresiones = CalcularCalidad(data_table, features=['58a','58b','58c','58d','58e','58f','58g','58h'])
    agresiones.rename('Agresiones', inplace=True)
    agresiones = NumberToTiemp_Bienes(agresiones)

    X[53] = FillSI_NO(X[53])
    X[53].fillna('NO', inplace=True)

    X['64as'] = NumberToTiemp_Bienes(X['64as'])
    X['64bs'] = NumberToTiemp_Bienes(X['64bs'])
    X['64cs'] = NumberToTiemp_Bienes(X['64cs'])
    X['64ds'] = NumberToTiemp_Bienes(X['64ds'])
    X['64es'] = NumberToTiemp_Bienes(X['64es'])
    X['64fs'] = NumberToTiemp_Bienes(X['64fs'])

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

    X['31a'] = NumberToTiemp_Bienes(X['31a'])
    X['31b'] = NumberToTiemp_Bienes(X['31b'])
    X['31c'] = NumberToTiemp_Bienes(X['31c'])

    X['32a'] = NumberToImportancia(X['32a'])
    X['32b'] = NumberToImportancia(X['32b'])
    X['32c'] = NumberToImportancia(X['32c'])
    X['32d'] = NumberToImportancia(X['32d'])
    X['32e'] = NumberToImportancia(X['32e'])
    SinglePlot(X['32a'], ' ', '# de Jovenes', 'Estudia Actualmente?')
    #X['31a'] =
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
    X['27a'] = NumberToCalidad_Bienes(X['27a'])
    X['27b'] = NumberToCalidad_Bienes(X['27b'])
    X['27c'] = NumberToCalidad_Bienes(X['27c'])
    X['27d'] = NumberToCalidad_Bienes(X['27d'])
    X['28a'] = NumberToCalidad_Bienes(X['28a'])
    X['28b'] = NumberToCalidad_Bienes(X['28b'])
    X['28c'] = NumberToCalidad_Bienes(X['28c'])
    X['28d'] = NumberToCalidad_Bienes(X['28d'])
    X['28e'] = NumberToCalidad_Bienes(X['28e'])
    X['28f'] = NumberToCalidad_Bienes(X['28f'])
    X['28g'] = NumberToCalidad_Bienes(X['28g'])
    X['29a'] = NumberToCalidad_Bienes(X['29a'])
    X['29b'] = NumberToCalidad_Bienes(X['29b'])
    X['29c'] = NumberToCalidad_Bienes(X['29c'])
    X['29d'] = NumberToCalidad_Bienes(X['29d'])

    X['36a'] = FillSI_NO(X['36a'])
    X['36b'] = FillSI_NO(X['36b'])

    X['49e'] = NumberToTiemp_Bienes(X['49e'])
    X['49e'] = NumberToTiemp_Bienes(X['49e'])
    X['49e'] = NumberToTiemp_Bienes(X['49e'])
    X['64gs'] = NumberToTiemp_Bienes(X['64gs'])
    #SinglePlot(X['64gs'], xTitle='', yTitle='Frecuencia', Title='Te Respetan?')

    bienes_servicios = CodificarEtiqueta(X, feature_names)
    pc_bienes, var_trans_bienes = ProcessPCA(bienes_servicios)

    return pc_bienes, var_trans_bienes

def NumberToImportancia(df):
    df.replace(1, 'Nada Importante',       inplace=True)
    df.replace(2, 'Poco Importante',       inplace=True)
    df.replace(3, 'Indiferente',           inplace=True)
    df.replace(4, 'Importante',            inplace=True)
    df.replace(5, 'Muy Importante',        inplace=True)
    df.replace(0, 'No conoce',             inplace=True)

    return df

def NumberToVeces_Convivencia(df):
    df.replace(1, 'Varias Veces', inplace=True)
    df.replace(2, 'Una vez',      inplace=True)
    df.replace(3, 'Nunca',        inplace=True)

    return df

def NumberToTiemp_Bienes(df):
    df.replace(1, 'Nunca',                   inplace=True)
    df.replace(2, 'Casi Nunca',              inplace=True)
    df.replace(3, 'Aveces',                  inplace=True)
    df.replace(4, 'Casi Siempre',            inplace=True)
    df.replace(5, 'Siempre',                 inplace=True)
    df.replace(0, 'No Aplica',               inplace=True)

    return df

def NumberToCalidad_Bienes(df):
    df.replace(1, 'Totalmente de desacuerdo',       inplace=True)
    df.replace(2, 'En desacuerdo',                  inplace=True)
    df.replace(3, 'Ni de acuerdo Ni en desacuerdo', inplace=True)
    df.replace(4, 'De acuerdo',                     inplace=True)
    df.replace(5, 'Totalmente de acuerdo',          inplace=True)

    return df

def NumberToPersonas(df):
    df.replace(1, '5 o mas',    inplace=True)
    df.replace(2, '4 personas', inplace=True)
    df.replace(3, '3 personas', inplace=True)
    df.replace(4, '2 personas', inplace=True)
    df.replace(5, '1 persona',  inplace=True)

    return df

def CalcularIDJM_Familia(data_table):
    feature_names = [23,
                    '25a','25b','25c','25d',
                    '59a',
                    '64af','64bf','64cf','64df','64ef','64ff','64gf','64hf','64if','64jf','64mf','64lf']  # pregunta 21 parece ser informacion irrelevante, es resumida en la pregunta 22 (numero de personas que habitan la casa)
                                                                             # preguntas 24 c a f dependen de b, irrelevante en este estudio
    X = data_table[feature_names]
    #X['24a'].fillna('No value', inplace=True)
    #X['24b'].fillna('No value', inplace=True)
    X[23] = NumberToPersonas(X[23])
    X['25a'] = NumberToCalidad_Bienes(X['25a'])
    X['25b'] = NumberToCalidad_Bienes(X['25b'])
    X['25c'] = NumberToCalidad_Bienes(X['25c'])
    X['25d'] = NumberToCalidad_Bienes(X['25d'])
    SinglePlot(X['64af'], xTitle='', yTitle='Frecuencia', Title='Tu familia te Respeta?')

    respeto = CalcularCalidad(X, features=['64af','64bf','64cf','64ef','64ff','64gf','64hf','64if','64jf','64mf','64lf'])
    respeto = NumberToTiemp_Bienes(respeto)

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
    #SinglePlot(idioma, ' ', '# de Jovenes', 'Estudia Actualmente?')


    data_table['pre_11'] = FillSI_NO(data_table['pre_11'])
    # val_idioma = np.array(idioma)
    # trans_val_idioma = CodificarEtiqueta(val_idioma)
    # print(trans_val_idioma)

    data = pd.concat([idioma, data_table[feature_names]], axis=1)
    new_features = ['Idiomas',
                    'pre_11',
                     '12a','12b','12c','12d','12e','12f','12g','12h','12i',
                     '13a','13b','13c','13d',
                     '14a','14b','14c','14d','14e','14f',
                     '59c',
                     '64ae','64be','64ce','64de','64ee','64fe','64ge','64he','64ie','64je','64me','64le']

    educacion    = CodificarEtiqueta(data, new_features)
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
    idioma.rename('Idiomas', inplace=True)
    #print(idioma.columns)
    idioma.apply(lambda x: idioma.replace(x, 'Poliglota', inplace=True) if x > 1 else x)
    idioma.replace(0, 'Solo Espanol',        inplace=True)
    idioma.replace(1, 'Bilingue',            inplace=True)

    return idioma

def CalcularIDJM_Trabajo(data_table):

    #feature_names = [16, 17, '18a', '18b', '18c','18d','19a','19b']  # preguntas 17 y 18 son codicionales a 17, no usar en el analisis
    feature_names = [16,'18a','18b','18c','18d','19a','19b','59b']
    X = data_table[feature_names]

    X['18a'] = NumberToAcuerdo_Desarrollo(X['18a'])
    X['18b'] = NumberToAcuerdo_Desarrollo(X['18b'])
    X['18c'] = NumberToAcuerdo_Desarrollo(X['18c'])
    X['18d'] = NumberToAcuerdo_Desarrollo(X['18d'])

    X['19a'].fillna(0, inplace=True)
    X['19b'].fillna(0, inplace=True)

    X['19a'] = NumberToAcuerdo_Desarrollo(X['19a'])
    X['19b'] = NumberToAcuerdo_Desarrollo(X['19b'])

    X['59b'] = NumberToTiemp_Bienes(X['59b'])
    #SinglePlot(X['18a'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')

    # SinglePlot(X['18a'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')
    # SinglePlot(X['18b'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')
    # SinglePlot(X['18c'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')
    # SinglePlot(X['18d'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')
    # SinglePlot(X['19a'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')
    # SinglePlot(X['19b'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')
    # SinglePlot(X['59b'], xTitle='Calificacion', yTitle='Frecuencia', Title='Habilidad Trabajo')



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

    primera_comp = pca.components_[0]
    # print('Forma: ' , pca_dataset.shape)
    # print('Primera Comp: ' , pca.components_[0])
    # print('Ratio variance: ', pca.explained_variance_ratio_)
    return primera_comp, pca_dataset

def CodificarEtiqueta(array, col):
    import category_encoders as ce

    print(type(array))
    encoder = ce.backward_difference.BackwardDifferenceEncoder(cols=col)
    encoder.fit(array, verbose=1)

    #print(type(std_data))

    ret_df = encoder.fit_transform(array)
    #print(array)
    # print('Tansfomed', ret_df.shape)
    # values_trans = TransformValuesToMinimunZero(ret_df)

    return ret_df

def TransformValuesToMinimunZero(array):
    min = np.amin(array)
    print(min)
    new_values = array + abs(min)

    print(new_values)
    return new_values

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
