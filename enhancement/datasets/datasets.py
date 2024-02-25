import copy

""" 
Purpose of this file:
make: returns a class reference based on the dataset_spec['name']

This is the input to the function 

train_dataset:
  dataset:
    name: image-folder-basic
   
"""

datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls

    return decorator


def make(dataset_spec, args=None):
    #! dataset_spec = dataset_dict["dataset"] or dataset_dict["wrapper"], args={"dataset": dataset}
    # name: image-folder-basic
    # args:
    #   root_path_inp: /home/gpu/girish/dataset/OUTDOOR_RGB/low_light/1_25/png/cam1/adobe/train_split
    #   root_path_out: /home/gpu/girish/dataset/OUTDOOR_RGB/well_lit/png/cam1/adobe/train_split
    #   split_key: train
    #   #! prolly split_file : <path to the split file>
    #   repeat: 30
    #   patchify: False
    #   patch_size: 512

    # Here the arguments from the configuration files are being loaded
    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec["args"])
        dataset_args.update(args)
    else:
        dataset_args = dataset_spec["args"]
        
        #   root_path_inp: /home/gpu/girish/dataset/OUTDOOR_RGB/low_light/1_25/png/cam1/adobe/train_split
        #   root_path_out: /home/gpu/girish/dataset/OUTDOOR_RGB/well_lit/png/cam1/adobe/train_split
    dataset = datasets[dataset_spec["name"]](**dataset_args)


    # IMPORTANT: dataset is a class. A class being returned here
    # ImageFolderBasic - 3 functions -->
    # def __init
    # ImageFolder(rootpath) - Check
    # print('hola')
    return dataset
