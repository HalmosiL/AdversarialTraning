import glob
import os

class SingletonClass(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(SingletonClass, cls).__new__(cls)
    return cls.instance

class DataLoaderManager(SingletonClass):
  ID_GETER_IS_FREE = True
  TRAIN_QUEUE_USED = []
  VAL_QUEUE_USED = []
  
  def getID(self, data_queue_path, type_):
    if(not DataLoaderManager.ID_GETER_IS_FREE):
      return []
    
    DataLoaderManager.ID_GETER_IS_FREE = False
    queue = glob.glob(data_queue_path + "image_*")
    queue.sort()
    
    image_path = None
    label_path = None
    
    if(type_ == "train"):
      QUEUE_USED = DataLoaderManager.TRAIN_QUEUE_USED
    else:
      QUEUE_USED = DataLoaderManager.VAL_QUEUE_USED

    if(len(queue) != 0):
      for q in queue:
        q_int = int(q.split("_")[-1].split(".")[0])
        if(q_int is not in QUEUE_USED):
          QUEUE_USED.append(q_int)
          image_path = data_queue_path + "image_" + str(q_int) + ".pt"
          label_path = data_queue_path + "label_" + str(q_int) + ".pt"

    if(
        image_path is not None and
        os.path.exists(image_path) and
        os.path.exists(label_path)
    ):
      DataLoaderManager.ID_GETER_IS_FREE = True
      return [image_path, label_path]

    DataLoaderManager.ID_GETER_IS_FREE = True
    return []
  
