import json
import os

class SingletonClass(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(SingletonClass, cls).__new__(cls)
    return cls.instance

class DataLoaderManager(SingletonClass):
  ID_GETER_IS_FREE = True
  
  def getID(self, path_queue):
    if(not DataLoaderManager.ID_GETER_IS_FREE):
      return []
    
    DataLoaderManager.ID_GETER_IS_FREE = False
    
    with open(path_queue, 'r+') as f:
      data = json.load(f)
      
      if(len(data['IDS']) != 0):
          idx_ = data['IDS'][0]
          image_path = self.data_queue_path + "image_" + str(idx_) + ".pt"
          label_path = self.data_queue_path + "label_" + str(idx_) + ".pt"
          
          if(
              os.path.exists(image_path) and
              os.path.exists(label_path)
          ):
            print(data['IDS'])
            data['IDS'].pop(0)
            
            f.seek(0)
            json.dump(data, f)
            f.truncate()
            f.close()
            
            DataLoaderManager.ID_GETER_IS_FREE = True
            return [image_path, label_path]
            
      f.seek(0)
      json.dump(data, f)
      f.truncate()
      f.close()

      DataLoaderManager.ID_GETER_IS_FREE = True
      return []
  
