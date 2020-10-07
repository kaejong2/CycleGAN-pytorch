import argparse

class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Arguments for TreeGAN.')

        # Dataset arguments
        self._parser.add_argument('--dataset_path', type=str, default='./data/', help='Dataset file path.')
        self._parser.add_argument('--batch_size', type=int, default=32, help='Integer value for batch size.')
        self._parser.add_argument('--image_size', type=int, default=784, help='Integer value for number of points.')

        # Training arguments
        self._parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
        self._parser.add_argument('--num_epochs', type=int, default=300, help='Integer value for epochs.')
        self._parser.add_argument('--lr', type=float, default=1e-4, help='Float value for learning rate.')
        self._parser.add_argument('--ckpt_path', type=str, default='./save/checkpoints/', help='Checkpoint path.')
        self._parser.add_argument('--result_path', type=str, default='./save/generated/', help='Generated results path.')

        # Network arguments
        self._parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
        self._parser.add_argument('--latent', type=int, default=64, help='random latent size')
        self._parser.add_argument('--hidden', type=int, default=256, help='hidden layer size')
        
    def parser(self):
        return self._parser

