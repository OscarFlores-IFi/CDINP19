import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%% Importar los datos (leer la imagen)
img = mpimg.imread('../Data/indice.png')
plt.imshow(img)

#%% reordenar la imagen en una sola tabla 
d = img.shape
img_col = np.reshape(img,(d[0]*d[1],d[2]))

#%% Convertir los datos a media cero
media = img_col.mean(axis=0)
img_m = img_col - media

#%% obtener la matriz de covarianzas
img_cov = np.cov(img_m,rowvar=False)

#%% Obtener valores propios y vectores propios
w,v = np.linalg.eig(img_cov)

#%% Analizar los componentes principales
porcentaje = w/np.sum(w)

#%% Comprimir la imagen
componentes = w[0]
M_trans = np.reshape(v[:,0],(4,1))

img_new = np.matrix(img_m)*np.matrix(M_trans)

#%% Recuperar la imagen y visualizarla
img_recuperada = np.matrix(img_new)*np.matrix(M_trans.transpose())
img_recuperada = img_recuperada+media

img_r = img.copy()
for i in np.arange(4):
    img_r[:,:,i] = img_recuperada[:,i].reshape((d[0],d[1]))
    
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(img_r)