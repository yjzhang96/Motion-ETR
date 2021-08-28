import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    # import ipdb;ipdb.set_trace()
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset(opt)
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset(opt)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)
    print("dataset [%s] was created" % (dataset.name()))
    # dataset.initialize(opt)
    if opt.phase =='train':
        from data.val_dataset import ValDataset
        val_dataset = ValDataset(opt)
        return dataset, val_dataset
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def __init__(self, opt):
        super(CustomDatasetDataLoader,self).initialize(opt)
        print("Opt.nThreads = ", opt.nThreads)
        if opt.phase =='train':
            self.dataset, self.val_dataset = CreateDataset(opt)
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle= False,
                num_workers=int(opt.nThreads)
            )
        elif opt.phase == 'test':
            self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads)
        )
        

    def load_data(self):
        if self.opt.phase == 'train':
            return self.dataloader, self.val_dataloader
        if self.opt.phase == 'test':
            return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
