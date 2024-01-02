import numpy as np

file_path = './result/SC4822G0.npz'

x = np.load(file_path)
print(x['features'].shape)
print(x['labels'].shape)