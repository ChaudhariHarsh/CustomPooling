
import numpy as np

# Funnction For implementing Kinetic Pooling :

def KineticPool(image,layer_dim):
    (Nrow,Ncol) = image.shape
    
    prob = np.zeros((Nrow-layer_dim,Ncol-layer_dim),dtype = np.float)
    for col in range(Ncol-layer_dim):
        for row in range(Nrow-layer_dim):
            pixels = image[ row : row+layer_dim, col : col+layer_dim].flatten()
            prob[row,col] = np.sum((np.unique(pixels , return_counts=True)[1]/len(pixels))**2,dtype=np.float)
    
    return prob

#img = np.random.randint(9, size=(10, 10))
img = np.ones((10,10))
layer_dim = 3
print(img)
prob = KineticPool(img, layer_dim)
print(prob)
