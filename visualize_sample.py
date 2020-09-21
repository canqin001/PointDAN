from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d#from mpl_toolkits.mplot3d import Axes3D
from modelnet_dataloader import Modelnet40_data, Modelnet40_data_h5, Shapenet_data, Scannet_data_h5
import numpy as np
import os

def mkdir(path):
 
    folder = os.path.exists(path)
 
    if not folder:                   
        os.makedirs(path)            
        print("---  new folder...  ---")
        print("---  OK  ---")
 
    else:
        print("---  There is this folder!  ---")


def draw_pc(pc, show=False, save_dir=None):
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], marker='.')
    ax.grid(False)
    # ax.axis('off')
    if show:
        plt.show()
    if save_dir is not None:
        mkdir(save_dir)
        save_dir = save_dir + '/' + str(i) +'.jpg'
        plt.savefig(save_dir)
    plt.close()
for dataset in ['shapenet','scannet']:
    for state in ['train', 'test']:
        save_dir = '../3d_imgs/' + dataset + '/' + state
        mkdir(save_dir)
        class_num=0
        data_loader = Shapenet_data(pc_root='../PointDAN_Code/Transfer_3d_data/Transfer_3d_data/' + dataset, status=state)
        rand_list = np.random.permutation(len(data_loader))
        lable_list = ['Bathtub','Bed','Bookshelf','Cabinet','Chair','Keyboard','Lamp','Laptop','Sofa','Table']
        for i in rand_list: 
        	pc, lbl = data_loader.__getitem__(i)
        	for class_num in range(10):
        		print(lable_list[class_num],i) if lbl==class_num else None
        		draw_pc(pc.squeeze().transpose(1,0), save_dir = save_dir+'/'+ lable_list[class_num] ) if lbl==class_num else None
