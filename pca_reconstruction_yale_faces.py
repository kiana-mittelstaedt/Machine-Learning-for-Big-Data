'''
CSE 547, HW2, Q1
Kiana Mittelstaedt
This code runs in full in a Jupyter Notebook. Please change the path to the folder
where you have 'faces.csv' saved so that the file is read in correctly. Also, the 
figures will export to the directory that you specify in the os.chdir() line below.
'''

# import libraries 
import pandas as pd 
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy.linalg as linalg
from scipy.linalg import eigh

os.chdir('C:\\Users\\kiana\\Documents\\cse')

# read in data 
# each row is a single image flattened into a vector in a column-major order 

# reading it in with pandas forced the first row to be read in as the header 
#file = pd.read_csv(r"C:\Users\kiana\Documents\cse\faces.csv")
#tmp = file.to_numpy()
#yale_faces = np.transpose(tmp)

file = np.loadtxt('C:/Users/kiana/Documents/cse/faces.csv',delimiter=',')
yale_faces = np.transpose(file)

# define sigma 
S = 1/yale_faces.shape[1] * (yale_faces.dot(np.transpose(yale_faces)))

# from Ed - this makes it symmetric 
sigma = 1/2 * (S + np.transpose(S))

# eigendecomposition
# someone on Ed suggested that the scipy eigh is faster than the numpy eig
vals,vecs = eigh(sigma)

for i in [1,2,10,30,50]:
    print('Eigenvalue ' + str(i) + ': ' + str(vals[-i]))

# trace = sum of all eigenvalues
trace = np.trace(sigma)
print('Trace of sigma: ' + str(trace))

# fractional reconstruction error 
# plot the error for the first 50 values of k, i.e., 1 < k < 50
# x-axis is k, y-axis is the fractional reconstruction error 

k_range = list(range(1,51))
error = []
evalues_50 = vals[-50:]

for i in list(range(1,51)):
    tmp_error = 1 - (sum(evalues_50[-i:])/trace)
    error.append(tmp_error)

plt.plot(k_range,error)
plt.ylabel('Fractional Reconstruction Error')
plt.xlabel('k')
plt.savefig("reconstruction_error.png")

real_vecs = vecs.real

# plot eigenfaces 
# somehow couldn't figure out how to get this to work in a loop...
# brute force way

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(251)
ax.imshow(real_vecs.T[-1].reshape(84,96).T,cmap='gray')
ax.title.set_text('Eigenvector 1')
ax2 = fig.add_subplot(252)
ax2.imshow(real_vecs.T[-2].reshape(84,96).T,cmap='gray')
ax2.title.set_text('Eigenvector 2')
ax3 = fig.add_subplot(253)
ax3.imshow(real_vecs.T[-3].reshape(84,96).T,cmap='gray')
ax3.title.set_text('Eigenvector 3')
ax4 = fig.add_subplot(254)
ax4.imshow(real_vecs.T[-4].reshape(84,96).T,cmap='gray')
ax4.title.set_text('Eigenvector 4')
ax5 = fig.add_subplot(255)
ax5.imshow(real_vecs.T[-5].reshape(84,96).T,cmap='gray')
ax5.title.set_text('Eigenvector 5')
ax6 = fig.add_subplot(256)
ax6.imshow(real_vecs.T[-6].reshape(84,96).T,cmap='gray')
ax6.title.set_text('Eigenvector 6')
ax7 = fig.add_subplot(257)
ax7.imshow(real_vecs.T[-7].reshape(84,96).T,cmap='gray')
ax7.title.set_text('Eigenvector 7')
ax8 = fig.add_subplot(258)
ax8.imshow(real_vecs.T[-8].reshape(84,96).T,cmap='gray')
ax8.title.set_text('Eigenvector 8')
ax9 = fig.add_subplot(259)
ax9.imshow(real_vecs.T[-9].reshape(84,96).T,cmap='gray')
ax9.title.set_text('Eigenvector 9')
ax10 = fig.add_subplot(2, 5, 10)
ax10.imshow(real_vecs.T[-10].reshape(84,96).T,cmap='gray')
ax10.title.set_text('Eigenvector 10')
plt.subplots_adjust(top=0.4, bottom=0.01, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.savefig("eigenfaces.png")

# plot reconstructions
image_row = [0,23,64,67,256]
num_vecs = [0,1,2,5,10,50]

fig,axs = plt.subplots(5,6,figsize=(15,15))
for j in range(len(image_row)):
    for i in range(len(num_vecs)):
        k=num_vecs[i]
        img=image_row[j]
        if k == 0:
            recon = file[img].reshape(84,96).T
            axs[j,i].set_title('Face ' + str(img+1) + ': ' + 'Original')
        else:
            utu = real_vecs[:,-k:].dot(np.transpose(real_vecs[:,-k:]))
            recon = (file[img].dot(utu)).reshape(84,96).T
            axs[j,i].set_title('k=' + str(num_vecs[j]))
        axs[j,i].imshow(recon, cmap='gray')

plt.savefig("reconstructed_faces.png")

