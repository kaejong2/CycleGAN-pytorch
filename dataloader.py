import torch
import torchvision
import torchvision.transforms as transforms

def data_loader(args):
    # Image processing
    transform = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,),(0.5,))])

    # MNIST dataset
    mnist = torchvision.datasets.MNIST(root=args.root_path,
                                    train=True,
                                    transform=transform,
                                    download=True)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                            batch_size=args.batch_size, 
                                            shuffle=True)

    return data_loader