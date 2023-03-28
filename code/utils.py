import torch
import os

def save_checkpoint(net, acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')

def write_txt(labels, file_name):
    
    with open(file_name, 'w') as file:
        for label in labels[:,:]:
            file.write(' '.join(str(int(i)) for i in label[:]) + '\n')

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    
    parser.add_argument('--load_checkpoint', action='store_true',
                        help='Load checkpoint file if exist')
    parser.add_argument("--epoch", type=int, default=30,
                        help='Total epoch number for training')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='Batch size for data loader')
    parser.add_argument("--freeze_layer", type=int, default=100,
                        help='How many layers to freeze in ResNet50')
    
    return parser