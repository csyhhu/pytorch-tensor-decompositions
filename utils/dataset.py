"""
An utils code for loading dataset
"""
import os

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from datetime import datetime
import utils.imagenet_utils as imagenet_utils
import utils.CIFAR10_utils as CIFAR10_utils
# import utils.visda_utils as visda_utils
# import utils.office_utils as office_utils
# import utils.MNIST_utils as MNIST_utils
import utils.SVHN_utils as SVHN_utils
# import utils.CLEF_utils as CLEF_utils
# import utils.imagenet32_utils as imagenet32_utils
import utils.STL9_utils as STL9_utils
import utils.CIFAR9_utils as CIFAR9_utils


def get_mean_and_std(dataset, n_channels=3):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(n_channels):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_dataloader(dataset_name, split, batch_size, model_name = 'AlexNet', \
                   add_split = None, shuffle = True, ratio=-1, resample=False):

    print ('[%s] Loading %s-%s from %s' %(datetime.now(), split, add_split, dataset_name))

    if dataset_name == 'MNIST':

        data_root_list = ['/home/shangyu/MNIST', '/data/datasets/MNIST']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        # print(data_root)

        normalize = transforms.Normalize((0.1307,), (0.3081,))
        # if split == 'train':
        MNIST_transform =transforms.Compose([
                               # transforms.Resize(32),
                               transforms.ToTensor(),
                               normalize
                           ])
        # else:
        '''
        dataset = MNIST_utils.MNIST(data_root,
                                    train = True if split =='train' else False, add_split=add_split,
                                    download=True, transform=MNIST_transform, ratio=ratio)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        '''

    elif dataset_name == 'SVHN':

        data_root_list = ['/home/shangyu/SVHN', '/data/datasets/SVHN']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                print('Data root found: %s' %data_root)
                break

        # normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
        normalize = transforms.Normalize((0.4433,), (0.1192,))
        if split == 'train':
            trainset = SVHN_utils.SVHN(root=data_root, split='train', add_split=add_split, download=False,
                                     transform=transforms.Compose([
                                         transforms.Grayscale(),
                                         # transforms.RandomCrop(28, padding=4),
                                         transforms.Resize(28),
                                         # transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize
                                         ]), ratio=ratio)
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        elif split == 'test':
            trainset = SVHN_utils.SVHN(root=data_root, split='test', download=False,
                                     transform=transforms.Compose([
                                         transforms.Grayscale(),
                                         transforms.Resize(28),
                                         transforms.ToTensor(),
                                         normalize
                                         ]))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        # return loader

    elif dataset_name == 'CIFAR10':

        data_root_list = ['/home/shangyu/CIFAR10', '/home/sinno/csy/CIFAR10', '/data/datasets/CIFAR10']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        if split == 'train':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            # trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False,
                                                    # transform=transform_train)
            trainset = CIFAR10_utils.CIFAR10(root=data_root, train=True, download=True,
                                                    transform=transform_train, ratio=ratio)
            print ('Number of training instances used: %d' %(len(trainset)))
            loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

        elif split == 'test' or split == 'val':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True,
                                                   transform=transform_test)
            loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


    elif dataset_name == 'STL10':

        data_root_list = ['/data/datasets/STL10']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        if split == 'train':

            loader = torch.utils.data.DataLoader(
                datasets.STL10(
                    root=data_root, split='train', download=True,
                    transform=transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(96),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
                batch_size=batch_size, shuffle=True)

        if split in ['test', 'val']:
            loader = torch.utils.data.DataLoader(
                datasets.STL10(
                    root=data_root, split='test', download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])),
                batch_size=batch_size, shuffle=False)

    elif dataset_name == 'CIFAR9':

        data_root_list = ['/home/shangyu/datasets/CIFAR9', '/data/datasets/CIFAR9']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        if split == 'train':
            transform = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        loader = torch.utils.data.DataLoader(
            CIFAR9_utils.CIFAR9(root=data_root, split=split, transform=transform),
             batch_size=batch_size, shuffle=shuffle, num_workers = 2)

        return loader


    elif dataset_name == 'STL9':

        data_root_list = ['/home/shangyu/datasets/STL9', '/data/datasets/STL9']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break
        # tensor([0.4330, 0.4331, 0.4330])
        # tensor([0.2327, 0.2327, 0.2327])

        if split == 'train':
            transform = transforms.Compose([
                        transforms.Resize(36),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Normalize((0.4330, 0.4331, 0.4330), (0.2327, 0.2327, 0.2327)),
                    ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Normalize((0.4330, 0.4331, 0.4330), (0.2327, 0.2327, 0.2327)),
            ])

        loader = torch.utils.data.DataLoader(
            STL9_utils.STL9(root=data_root, split=split, transform=transform),
             batch_size=batch_size, shuffle=shuffle, num_workers = 2)

        return loader


    elif dataset_name == 'ImageNet':
        data_root_list = ['/remote-imagenet', '/data/imagenet', '/mnt/public/imagenet']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break
        traindir = ('../train_imagenet_list.pkl', '../classes.pkl', '../classes-to-idx.pkl','%s/train' %data_root)
        valdir = ('../val_imagenet_list.pkl', '../classes.pkl', '../classes-to-idx.pkl', '%s/val-pytorch' %data_root)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if split == 'train':
            trainDataset = imagenet_utils.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), ratio=ratio)
            print ('Number of training data used: %d' %(len(trainDataset)))
            loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, \
                                                 num_workers=4 * torch.cuda.device_count(), pin_memory=True)

        elif split == 'val' or split == 'test':
            valDataset = imagenet_utils.ImageFolder(valdir, transforms.Compose([
                # transforms.CenterCrop(32),
		        # transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
            loader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size, shuffle=True, \
                                                 num_workers=4 * torch.cuda.device_count(), pin_memory=True)

    elif dataset_name == 'ImageNet32':
        data_root_list = ['/data/ImageNet32', '/mnt/data/ImageNet32']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=[0.4811, 0.4575, 0.4078], std=[0.2051, 0.2010, 0.2022])
        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif split == 'test' or split == 'val':
            transform = transforms.Compose([
                # transforms.Resize(256),
                # transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        '''
        dataset = imagenet32_utils.ImageNet32(root=data_root, train = True if split == 'train' else False,
                                              transform=transform, ratio=ratio)
        print('Number of training instances used: %d' % (len(dataset)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        '''

    elif dataset_name == 'visda':
        data_root_list = ['/data/visda2017', '/home/shangyu/visda2017']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break

        if split == 'train':
            normalize = transforms.Normalize(mean=[0.8197, 0.8175, 0.8129], std=[0.1932, 0.1955, 0.2007])
        elif split == 'validation':
            normalize = transforms.Normalize(mean=[0.4094, 0.3891, 0.3699], std=[0.2189, 0.2131, 0.2103])
        else:
            normalize = transforms.Normalize(mean=[0.4098, 0.3894, 0.3702], std=[0.2188, 0.2131, 0.2103])

        if add_split is None:
            if model_name == 'AlexNet':
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])

        else:
            if add_split in ['train', 'label'] and model_name == 'AlexNet':
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif add_split in ['test'] and model_name == 'AlexNet':
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif add_split in ['train', 'label'] and model_name != 'AlexNet':
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif add_split in ['test'] and model_name != 'AlexNet':
                transform = transforms.Compose([
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                print('Undefine split method')
                # os._exit(0)
        '''
        dataset = visda_utils.visda_dataset(root=data_root,split=split, \
                                            transform=transform, add_split=add_split, ratio=ratio)
        loader = torch.utils.data.DataLoader(dataset, \
                     batch_size=batch_size, shuffle=shuffle,\
                     num_workers=4 * torch.cuda.device_count(), pin_memory=True)
        print('Batch number of %s data used from visda-%s: %d' % (add_split, split, len(loader)))
        '''

    elif dataset_name == 'office':
        if split == 'amazon':
            normalize = transforms.Normalize(mean=[0.7924, 0.7862, 0.7842], std=[0.2749, 0.2796, 0.2809])
        elif split == 'webcam':
            normalize = transforms.Normalize(mean=[0.6120, 0.6188, 0.6173], std=[0.2259, 0.2318, 0.2355])
        elif split == 'dslr':
            normalize = transforms.Normalize(mean=[0.4709, 0.4487, 0.4064], std=[0.1824, 0.1759, 0.1788])

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if add_split is None or add_split in ['train', 'label']:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        data_root_list = ['/data/office', '/home/shangyu/office']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break
        '''
        dataset = office_utils.office_dataset(root=data_root, split=split, \
                                            transform=transform, add_split=add_split, ratio=ratio, resample=resample)
        loader = torch.utils.data.DataLoader(dataset, \
                                             batch_size=batch_size, shuffle=shuffle, \
                                             num_workers=4 * torch.cuda.device_count(), pin_memory=True)
        print('Number of images load from %s-%s in %s: %d' %(split, add_split, dataset_name, len(dataset)))
        print('Number of batches: %d' %len(loader))
        '''

    elif dataset_name == 'Image_CLEF':
        data_root_list = ['/data/image_CLEF', '/home/shangyu/Image_CLEF']
        for data_root in data_root_list:
            if os.path.exists(data_root):
                break
        if split == 'b':
            normalize = transforms.Normalize(mean=[0.5246, 0.5139, 0.4830], std=[0.2505, 0.2473, 0.2532])
        elif split == 'c':
            normalize = transforms.Normalize(mean=[0.5327, 0.5246, 0.4938], std=[0.2545, 0.2505, 0.2521])
        elif split == 'i':
            normalize = transforms.Normalize(mean=[0.4688, 0.4593, 0.4328], std=[0.2508, 0.2463, 0.2508])
        elif split == 'p':
            normalize = transforms.Normalize(mean=[0.4634, 0.4557, 0.4319], std=[0.2355, 0.2330, 0.2397])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if add_split is None or add_split in ['train', 'label', 'unlabel']:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        '''
        dataset = CLEF_utils.CLEF_dataset(root=data_root, split=split, \
                                              transform=transform, add_split=add_split,
                                              ratio=ratio, resample=resample)
        loader = torch.utils.data.DataLoader(dataset, \
                                             batch_size=batch_size, shuffle=shuffle, \
                                             num_workers=4 * torch.cuda.device_count(), pin_memory=True)
        '''

    print ('[DATA LOADING] Loading from %s-%s-%s finish. Number of images: %d, Number of batches: %d' \
           %(dataset_name, split, add_split, len(loader.dataset), len(loader)))

    return loader


if __name__ == '__main__':

    # stl9 = get_dataloader('STL9', 'train', 128)
    cifar9 = get_dataloader('CIFAR9', 'train', 128)
    # cifar10 = get_dataloader('CIFAR10', 'train', 128)

    # for inputs, targets in cifar10:
    for inputs, targets in cifar9:
    # for inputs, targets in stl9:
        print(inputs.shape)
        # print(targets)
        # input()
        break