import os
import sys
import numpy as np
from skimage import io
image_num = 415
ig = []

def average_face(x):

    total = np.zeros(1080000)
    for e in x:
    	total+=e
    mean = total / image_num
    io.imsave('face_mean.jpg',mean.reshape(600,600,3).astype('uint8'))   

def eigen_face(x):
    x = x.reshape((415, 360000*3))
    
    ave = np.mean(x,axis=0)

    m,n = np.shape(x)
 
    x = x-ave

    u,s,v = np.linalg.svd(x.transpose(),full_matrices=False)
    
    fig = plt.figure(figsize=(4,4))
    for i in range(4):
        ax = fig.add_subplot(2,2,i+1)
        E = -1*u[:,i].reshape(600,600,3)
        E-= np.min(E)                          
        E/= np.max(E)
        E = (E * 255).astype(np.uint8)
        ax.imshow(E,cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

    fig.savefig('eigenface.png')

def recon_face(x):
    ave = np.mean(x,axis=0)
    x_ = x - ave
    #(415,1080000)
    k = 4
    u,s,v = np.linalg.svd(x_.T,full_matrices=False)    #==> u : eigenvector    s : eigenvalue
    #u:(1080000, 415),s:(415,),v:(415, 415)
    #path = os.path.join(sys.argv[1],sys.argv[2])   ##做報告
    #img = io.imread(path)                          ##讀檔改這兩行
    img = io.imread(os.path.join(sys.argv[1],sys.argv[2]))
    img = img.flatten()
    img = img - ave
    w = np.dot(img,u)#got 4 weight
    recon = np.dot(w[:k],u[:,:k].T)#1080000
    recon = recon + ave

    recon-= np.min(recon)
    recon/= np.max(recon)
    recon = (recon*255).astype(np.uint8)
    io.imsave('reconstruction.jpg',recon.reshape(600,600,3).astype('uint8'))


def calculate_evr(x):  #找s
    
    ave = np.mean(x,axis=0)
    x_ = x - ave
    #(415,1080000)
    k = 415
    p = 4
    u,s,v = np.linalg.svd(x_.T,full_matrices=False)
    #u:(1080000, 415),s:(415,),v:(415, 415)
    
    s_ratio = s / np.sum(s)

    print('%.4f,%.4f,%.4f,%.4f'%tuple(s_ratio[:4]))
    print('%.1f,%.1f,%.1f,%.1f'%tuple(s_ratio[:4]*100))

def load_image(img,i):

    current_ig = io.imread(img)
    ig.append(current_ig.flatten())
    
for i in range(image_num):
    image_name = os.path.join(sys.argv[1],'%d.jpg' %i)
    if i%10 ==1:
        s = 'st';
    elif i %10 ==2:
        s = 'nd';
    elif i %10 ==3:
        s = 'rd';
    else:
        s = 'th'
    print(i)
    load_image(image_name,i)

ig = np.asarray(ig)
#average_face(ig)       #problem 1.1       
#s = eigen_face(ig)     #problem 1.2
recon_face(ig)         #problem 1.3
#calculate_evr(ig)       #problem 1.4