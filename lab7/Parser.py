import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.5, type=float, help='momentum term for adam')
    parser.add_argument('--epoch', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--latent_size', default=100, type=int, help='latent dimension')
    parser.add_argument('--condition_size', default=200, type=int, help='condition dimension')
    parser.add_argument('--seed', default=9899, type=int, help='manual seed')

    # wgan
    parser.add_argument('--use_wgan', default=False, action='store_true') 
    parser.add_argument('--clamp_num', default=0.01, type=float, help='clamp_num')

    #acgan
    parser.add_argument('--lambda_cls', default=5, type=float, help='lambda_cls')

    parser.add_argument('--log_root', default='./logs/', help='root directory for logs')
    parser.add_argument('--load_weight', default=False, action='store_true')  
    parser.add_argument('--weight_root', default='./', help='root directory for logs')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--test_file', default='./test.json', help='test file name')

    args = parser.parse_args()
    return args