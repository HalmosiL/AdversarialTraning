from brain_plasma import Brain
import os

class SingletonClass(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(SingletonClass, cls).__new__(cls)
    return cls.instance

class DataLoaderManager(SingletonClass):
  ID_GETER_IS_FREE = True
  BRAIN = Brain()
  
  def getID(self, data_queue_path, type_):
    if(not DataLoaderManager.ID_GETER_IS_FREE):
      return []
    
    DataLoaderManager.ID_GETER_IS_FREE = False

    if(type_ == "train"):
      data = DataLoaderManager.BRAIN["train_queue"]
    else:
      data = DataLoaderManager.BRAIN["val_queue"]
    
    if(len(data) != 0):
        image_path = data_queue_path + "image_" + str(data[0]) + ".pt"
        label_path = data_queue_path + "label_" + str(data[0]) + ".pt"

        if(
            os.path.exists(image_path) and
            os.path.exists(label_path)
        ):
          data.pop(0)
          DataLoaderManager.ID_GETER_IS_FREE = True
          return [image_path, label_path]

    DataLoaderManager.ID_GETER_IS_FREE = True
    return []
  
