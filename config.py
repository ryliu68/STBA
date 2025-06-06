# now 2024.03
attack_configs = {
    'clean': {
        'CIFAR-10': {'n_pop': 10, 'sigma': 0.08, 'lr': 0.025, 'flow_bounds': [2, 5], 'adjust_num': 30},
        'CIFAR-100': {'n_pop': 10, 'sigma': 0.08, 'lr': 0.025, 'flow_bounds': [2, 3], 'adjust_num': 30},
        'STL-10': {'n_pop': 20, 'sigma': 0.1, 'lr': 0.08, 'flow_bounds': [2, 5], 'adjust_num': 20},
        'ImageNet': {'n_pop': 20, 'sigma': 0.05, 'lr': 0.04, 'flow_bounds': [2, 5], 'adjust_num': 10}},

    'defense': {
        'CIFAR-10': {'n_pop': 10, 'sigma': 0.08, 'lr': 0.025, 'flow_bounds': [0.5, 2], 'adjust_num': 20},
        'CIFAR-100': {'n_pop': 20, 'sigma': 0.1, 'lr': 0.08, 'flow_bounds': [1, 3], 'adjust_num': 10},
        'STL-10': {'n_pop': 20, 'sigma': 0.1, 'lr': 0.08, 'flow_bounds': [2, 5], 'adjust_num': 10},
        'ImageNet': {'n_pop': 10, 'sigma': 0.2, 'lr': 0.08, 'flow_bounds': [2, 5], 'adjust_num': 10}},
}


def set_attack_config(args):
    args.n_pop = attack_configs[args.target_model][args.dataset]['n_pop']
    args.sigma = attack_configs[args.target_model][args.dataset]['sigma']
    args.lr = attack_configs[args.target_model][args.dataset]['lr']
    args.flow_bounds = attack_configs[args.target_model][args.dataset]['flow_bounds']
    args.adjust_num = attack_configs[args.target_model][args.dataset]['adjust_num']
