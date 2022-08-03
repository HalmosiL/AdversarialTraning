from filelock import FileLock

import json
import os

class SingletonClass(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(SingletonClass, cls).__new__(cls)
    return cls.instance

class DataLoaderManager(SingletonClass):
  ID_GETER_IS_FREE = True
  
  def getID(self, path_queue, data_queue_path):
    if(not DataLoaderManager.ID_GETER_IS_FREE):
      return []
    
    DataLoaderManager.ID_GETER_IS_FREE = False
    
    with FileLock(path_queue):
      with open(path_queue, 'r+') as f:
        data = json.load(f)

        if(len(data['IDS']) != 0):
            image_path = data_queue_path + "image_" + str(data['IDS'][0]) + ".pt"
            label_path = data_queue_path + "label_" + str(data['IDS'][0]) + ".pt"

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
  
