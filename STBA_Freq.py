import argparse
import warnings
import os
import numpy as np
import torch
from tqdm import tqdm

from config import set_attack_config
from model import load_clean_model
from utils import cw_loss, flow_st, Loss_flow, calc_Freq
from utils import distance_2 as distance, set_seed, seed_data_and_model
from torch.nn.functional import softmax

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")


# Set device to GPU if available, else CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LFFA(object):
    """
    LFFA: Latent Frequency Flow Attack.
    Implements a black-box adversarial attack based on frequency domain perturbation and NES optimization.
    """

    def __init__(self, attack_model, args):
        self.attack_model = attack_model
        self.target_model = args.target_model
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.field_size = args.field_size
        self.num_queries = 0
        self.flow_bound = 2
        self.device = device

    def get_probs(self, x):
        """
        Pass input x through the attack model and return the output probabilities.
        """
        output = self.attack_model(x)
        return output

    def check_success(self, x_high, x_low, mu, y):
        """
        Check whether the current adversarial example fools the model.
        Returns (True, adv_img) if attack is successful, else (False, None).
        """
        # Add perturbation to the high-frequency component and combine with low-frequency
        real_input_img = flow_st(
            x_high, torch.clamp(mu.view(self.field_size), -self.flow_bound, self.flow_bound)
        )
        real_input_img = real_input_img + x_low
        pred_adv = self.get_probs(real_input_img).argmax(dim=1)
        self.num_queries += x_high.size(0)
        # If the prediction changes, attack is successful
        if pred_adv != y:
            return True, real_input_img
        else:
            return False, None

    def lffa_attack(self, args, x_high, x_low, y, t=None):
        """
        Main attack routine using NES optimization in the frequency domain.
        """
        num_iters = int(args.max_queries / args.n_pop)
        adjust_step = num_iters / args.adjust_num
        idx_num = int(num_iters / adjust_step)

        with torch.no_grad():
            self.num_queries = 0
            self.flow_bound = args.flow_bounds[0]
            # Initialize mu (shift vector in latent space)
            mu = 0.1 * torch.rand(
                [1, self.field_size[1] * self.field_size[2] * self.field_size[3]]
            ).to(device)

            for i in tqdm(range(num_iters), disable=True):
                # Check attack success
                success_, x_adv = self.check_success(x_high, x_low, mu, y)
                if success_:
                    return x_adv, self.num_queries

                # Dynamically adjust the flow bound
                if (i + 1) % adjust_step == 0:
                    self.flow_bound += (
                        (args.flow_bounds[1] - args.flow_bounds[0]) / idx_num
                    )

                # NES: Sample Gaussian noise and update mu
                z_sample = torch.randn(
                    [args.n_pop, self.field_size[1] * self.field_size[2] * self.field_size[3]]
                ).to(device)
                modify_try = mu.repeat(args.n_pop, 1) + args.sigma * z_sample

                x_hat_s = flow_st(
                    x_high,
                    torch.clamp(
                        modify_try.view(
                            args.n_pop,
                            self.field_size[1],
                            self.field_size[2],
                            self.field_size[3],
                        ),
                        -self.flow_bound,
                        self.flow_bound,
                    ),
                )
                x_hat_s = x_hat_s + x_low

                if self.num_queries > args.max_queries:
                    return None, 0

                # Query the model and calculate loss
                outputs = self.get_probs(x_hat_s.squeeze())
                outputs = softmax(outputs, dim=1)
                self.num_queries += args.n_pop
                cw_l = cw_loss(args, outputs, y)

                flows = torch.clamp(
                    modify_try.view(
                        args.n_pop,
                        self.field_size[1],
                        self.field_size[2],
                        self.field_size[3],
                    ),
                    -self.flow_bound,
                    self.flow_bound,
                )

                # Regularization loss on flow
                f_loss = [Loss_flow()((flow).unsqueeze(0)) for flow in flows]
                f_loss = torch.stack(f_loss)
                loss1 = cw_l + 5 * f_loss

                # NES update for mu
                Reward = 0.5 * loss1
                A = (Reward - torch.mean(Reward)) / (torch.std(Reward) + 1e-10)
                mu -= (args.lr / (args.n_pop * args.sigma)) * (
                    torch.matmul(z_sample.view(args.n_pop, -1).t(), A.view(-1, 1))
                ).view(1, -1)

            return None, 0


def attack(args):
    """
    Main attack loop for a batch of images and a list of models.
    """
    if os.path.isfile(args.batchfile):
        checkpoint = torch.load(args.batchfile)
        images = checkpoint['images']
        labels = checkpoint['labels']
    else:
        raise NameError("can't find the data")

    for model_name in args.model_names:
        args.model = model_name
        model = load_clean_model(args)
        model = model.eval().to(device)
        attacker = LFFA(model, args)

        success = 0
        tested_images = 0
        total_queries = []
        total_l2 = 0
        total_linf = 0
        total_lpips = 0
        total_dists = 0
        total_ssim = 0

        total_batch = int(images.size(0) / args.batch_size)

        for i_batch in range(total_batch):
            if tested_images >= args.test_num:
                break
            x = images[i_batch * args.batch_size: (i_batch + 1) * args.batch_size].to(device)
            y = labels[i_batch * args.batch_size: (i_batch + 1) * args.batch_size].to(device)
            tested_images += x.size(0)

            # Split image into high- and low-frequency components
            x_high, x_low = calc_Freq(args, x)
            adv, nqueries = attacker.lffa_attack(args, x_high, x_low, y)

            if nqueries == 0:
                continue
            else:
                if success == 0:
                    clean_imgs = x.detach().cpu()
                    adv_imgs = adv.detach().cpu()
                    labels_ = y.detach().cpu()
                else:
                    clean_imgs = torch.vstack((clean_imgs, x.detach().cpu()))
                    adv_imgs = torch.vstack((adv_imgs, adv.detach().cpu()))
                    labels_ = torch.vstack((labels_, y.detach().cpu()))

            if nqueries != 0:
                success += 1
                total_queries.append(nqueries)
                total_l2 += distance(adv, x, norm="l2")
                total_linf += distance(adv, x, norm="linf")
                total_lpips += distance(adv, x, norm="lpips")
                total_dists += distance(adv, x, norm="dists")
                total_ssim += distance(adv, x, norm="ssim")  # 1-ssim

        print(
            F"\nVictim model: {args.model}\ttested_images: {tested_images}\ttotal_queries:\t{total_queries}"
        )
        log = F"\nASR\tAvg.Q\tMed.Q\tL2\tLinf\tlpips\tdists\tssim"
        print(log)

        # Calculate statistics
        ASR = round(success * 100 / tested_images, 2)
        Avg_Q = round(np.mean(total_queries), 2)
        Med_Q = round(np.median(total_queries), 2)
        Avg_l2 = round(total_l2 / success, 2)
        Avg_linf = round(total_linf / success, 2)
        Avg_lpips = round(total_lpips / success, 2)
        Avg_dists = round(total_dists / success, 2)
        Avg_ssim = round(total_ssim / success, 2)

        log = F"{ASR}\t{Avg_Q}\t{Med_Q}\t{Avg_l2}\t{Avg_linf}\t{Avg_lpips}\t{Avg_dists}\t{Avg_ssim}"
        print(log)

        # Save attack results
        state = {
            'adv_imgs': adv_imgs,
            'clean_imgs': clean_imgs,
            'labels': labels_,
        }
        # Set save path
        save_path = "adv"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(state, F"{save_path}/{args.dataset}_{args.model}_q{args.max_queries}_Freq.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for parallel runs')
    parser.add_argument('-D', '--dataset', type=str, default="ImageNet",
                        help='Dataset to be used, [CIFAR-10, CIFAR-100, STL-10, ImageNet]')
    parser.add_argument('--defense-model-dir', type=str, default='./model/defense_models', help='Directory for defense models')
    parser.add_argument('-T', '--target-model', default="clean", help='clean | defense')
    parser.add_argument('-Q', '--max-queries', type=int, default=1000, help='Maximum number of queries, 0 for unlimited')
    parser.add_argument('--test-num', type=int, default=1000, help='Maximum number of images to test')
    parser.add_argument('-S', '--seed', default=2023, type=int, help='Random seed')

    args = parser.parse_args()

    # Setup config and random seed
    set_attack_config(args)
    print(args)
    set_seed(args.seed)
    seed_data_and_model(args)

    # Run the attack
    attack(args)
