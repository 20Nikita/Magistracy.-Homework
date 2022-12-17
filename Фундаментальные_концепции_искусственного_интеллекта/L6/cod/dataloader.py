from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from cod.transform import get_ransforms
import cod.Dataset as Dataset

def dataloader(SIZE, dirr, file, status, datasetName, N_class, transforms = False, batch_size = 100, num_workers=0, pin_memory=True, drop_last=True, shuffle=False, rasp_file = False, t1=None, t2=None, t3=None):
    if not transforms:
        val_transforms, train_transforms = get_ransforms(SIZE)
    else:
        val_transforms, train_transforms = transforms(SIZE)
    sampler = None
    if not not rasp_file:
        rasp_file = open(dirr + "/" + rasp_file, "r")
        rasp_ist = rasp_file.readlines()
        rasp_ist = [float(rasp_ist[i]) for i in range(len(rasp_ist))]
        sampler = WeightedRandomSampler(rasp_ist, len(rasp_ist))
    if status == "train":
        transforms = train_transforms
        dataset = Dataset.__dict__[datasetName](dirr + "/" + file, dirr, N_class, transforms, t1, t2, 1)
    elif status == "val":
        transforms = val_transforms
        dataset = Dataset.__dict__[datasetName](dirr + "/" + file, dirr, N_class, transforms, t1, t2, 0)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,shuffle=shuffle,sampler=sampler)



