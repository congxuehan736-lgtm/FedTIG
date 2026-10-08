import argparse
import os
from Dataset.param_aug import ParamDiffAug

def args_parser():
    parser = argparse.ArgumentParser()

    path_dir = os.path.dirname(__file__)

    # general
    parser.add_argument('--algorithm', type=str, default='fedlf', choices=['creff', 'fedavg', 'fedprox', 'fedic','fedbn','focalloss','fedrs','fedlf'],
                        help='choice your algorithm')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--num_clients', type=int, default=40)
    parser.add_argument('--num_rounds', type=int, default=200, help='全局迭代次数')
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--num_epochs_local_training', type=int, default=5)
    parser.add_argument('--batch_size_local_training', type=int, default=32)
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_online_clients', type=int, default=16)
    parser.add_argument('--match_epoch', type=int, default=100)
    parser.add_argument('--crt_epoch', type=int, default=300)
    parser.add_argument('--batch_real', type=int, default=32)
    parser.add_argument('--num_of_feature', type=int, default=100)
    parser.add_argument('--lr_feature', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_size_test', type=int, default=500)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.02, type=float, help='imbalance factor')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--save_path', type=str, default=os.path.join(path_dir, 'result/'))
    parser.add_argument('--method', type=str, default='DSA', help='DC/DSA')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    # FedLF
    parser.add_argument('--lambda_2', type=float, default=0.5,help='lambda for senond loss')
    parser.add_argument('--warm_up_epoch', type=int, default=10, help='number of warm up')
    # FedIC
    parser.add_argument('--num_data_train', type=int, default=49000)
    parser.add_argument('--total_steps', type=int, default=100)
    parser.add_argument('--server_steps', type=int, default=100)
    parser.add_argument('--mini_batch_size', type=int, default=20)
    parser.add_argument('--mini_batch_size_unlabeled', type=int, default=128)
    parser.add_argument('--lr_global_teaching', type=float, default=0.001)
    parser.add_argument('--temperature', type=float, default=2)
    parser.add_argument('--ld', type=float, default=0.5)
    parser.add_argument('--ensemble_ld', type=float, default=0.0)

    # Focal_Loss
    parser.add_argument('--alpha', default=0.25, type=float, help='Focal Loss')
    parser.add_argument('--gamma', default=2.0, type=float, help='Focal Loss')
    # FedRS
    parser.add_argument('--rs_alpha', default=0.25, type=float, help='FedRS')
    
    # FedProx
    parser.add_argument('--mu', type=float, default=0.01)
    # FedAvgM
    parser.add_argument('--init_belta', type=float, default=0.97)

    # ========== 新增创新点参数 ==========
    # 客户端采样策略：uniform（均匀） 或 tail_weighted（按尾部类样本数加权）
    parser.add_argument('--sampling_strategy', type=str, default='uniform',
                        choices=['uniform', 'tail_weighted'],
                        help='Client sampling strategy')
    # 梯度重加权系数：对尾部类的分类器权重和偏置乘以该系数（1.0 表示不重加权）
    parser.add_argument('--grad_rew_weight', type=float, default=1.0,
                        help='Multiplier for tail class gradients in aggregation')

    args = parser.parse_args()

    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    return args