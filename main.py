import argparse
import torch
from train import cycleGAN
from test import test
def Arguments():
    parser = argparse.ArgumentParser(description='Arguments for CycleGAN.')

    parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
    parser.add_argument('--mode', type=str, default='train', choices=["train", "test"], help='Run type.')
    # Dataset arguments
    parser.add_argument('--batch_size', type=int, default=4, help='Integer value for batch size.')
    parser.add_argument('--image_size', type=int, default=256, help='Integer value for number of points.')
    parser.add_argument('--input_nc', type=int, default=3, help='size of image height')
    parser.add_argument('--output_nc', type=int, default=3, help='size of image height')
    parser.add_argument('--channels', type=int, default=10, help='Number of image channels')
    
    # Optimizer arguments
    parser.add_argument('--b1', type=int, default=0.5, help='GPU number to use.')
    parser.add_argument('--b2', type=int, default=0.999, help='GPU number to use.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Adam : learning rate.')
    parser.add_argument('--decay_epoch', type=int, default=100, help="epoch from which to start lr decay")

    # Training arguments
    parser.add_argument('--epoch', type=int, default=0, help='Epoch to start training from.')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs of training.')
    parser.add_argument('--save_freq', type=int, default=1, help='Number of save frequency.')
    parser.add_argument('--data_path', type=str, default='/mnt/hdd/jongjin/cyclegan/monet2photo/', help='Checkpoint path.')
    parser.add_argument('--ckpt_path', type=str, default='/mnt/hdd/jongjin/cyclegan/ckpt/', help='Checkpoint path.')
    parser.add_argument('--result_path', type=str, default='/mnt/hdd/jongjin/cyclegan/result/', help='Generated results path.')
    
    # Model arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = Arguments()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    for arg, value in vars(args).items():
        print("log %s : %r" % (arg, value))
    if args.mode == 'train':
        model = cycleGAN(args)
        model.run(ckpt_path=args.ckpt_path, result_path=args.result_path)
    elif args.mode == 'test':
        test(args)