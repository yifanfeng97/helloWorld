import h5py
import os.path as osp

def getDataFiles(list_filename):
  return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
  f = h5py.File(h5_filename)
  data = f['data'][:]
  label = f['label'][:]
  return (data, label)

def loadDataFile(filename):
  return load_h5(filename)


def load_h5_data_label_seg(h5_filename):
  f = h5py.File(h5_filename)
  data = f['data'][:] # (2048, 2048, 3)
  label = f['label'][:] # (2048, 1)
  faceId = f['faceId'][:] # (2048, 2048)
  return (data, label, faceId)

if __name__ == '__main__':
    data_root = '../../data/modelnet40_ply_hdf5_2048'
    file_name = osp.join(data_root, 'ply_data_train4.h5')
    if osp.exists(file_name):
        data, label, norm = load_h5_data_label_seg(file_name)
        print(len(data))