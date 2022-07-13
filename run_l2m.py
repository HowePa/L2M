from pathlib import Path
import argparse
from time import time
import pandas as pd

import torch
from torch.utils.data import DataLoader
import dgl

from Datasets.utils import get_dataset
from L2M.ppo.framework import ProxPolicyOptimFramework
from L2M.ppo.actor_critic import ActorCritic
from L2M.ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet
from L2M.ppo.storage import RolloutStorage
from L2M.env import MaximumMatchingEnv
from EEN.message_pass import sum_efeat, max_efeat

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, default='train', choices=['train', 'test'])
parser.add_argument("--type", type=str, default='er', choices=['er', 'ba', 'ws', 'hk', 'real'])
parser.add_argument("--dataset", type=str, default=None)  # only real dataset need

parser.add_argument("--minn", type=int, default=50)
parser.add_argument("--maxn", type=int, default=100)

parser.add_argument("--er_p", type=float, default=0.15)
parser.add_argument("--ws_p", type=float, default=0.15)
parser.add_argument("--hk_p", type=float, default=0.05)
parser.add_argument("--k", type=float, default=4)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# actor critic
num_layers = 4
input_dim = 2
output_dim = 2
hidden_dim = 128

# optimization
init_lr = 1e-4
max_epi_t = 32
max_rollout_t =32
max_update_t = 20000

# ppo
gamma = 1.0
clip_value = 0.2
optim_num_samples = 4
critic_loss_coef = 0.5
reg_coef = 0.1
max_grad_norm = 0.5

# logging
log_freq = 20
vali_freq = 200
save_freq = 5000

# main
rollout_batch_size = 32
eval_batch_size = 32
optim_batch_size = 16
train_num_samples = 1
eval_num_samples = 1

# dataset
Model_Path = Path("models/l2m")
if not Model_Path.exists():
    Model_Path.mkdir(parents=True, exist_ok=True)
dataset_args = {}
dataset_args['type'] = args.type
if args.type == 'real':
    dataset_args['data_dir'] = args.dataset
    Model_Name = "{}.pt".format(args.dataset)
else:
    dataset_args['min_n'] = args.minn
    dataset_args['max_n'] = args.maxn
    Model_Name = "{}_{}_{}".format(args.type, args.minn, args.maxn)
    if args.type == 'er':
        dataset_args['er_p'] = args.er_p
        Model_Name = Model_Name + "_{}".format(args.er_p)


def collate_fn(graphs):
    return dgl.batch(graphs)


env = MaximumMatchingEnv(max_epi_t=max_epi_t, device=device)
actor_critic = ActorCritic(actor_class=PolicyGraphConvNet,
                           critic_class=ValueGraphConvNet,
                           input_dim=input_dim,
                           hidden_dim=hidden_dim,
                           output_dim=output_dim,
                           num_layers=num_layers,
                           device=device)


def evaluate(mode, actor_critic):
    actor_critic.eval()
    cum_cnt = 0
    cum_eval_sol = 0.0
    T = 0.0
    n_rounds = 0
    num_steps = 0
    for g in data_loaders[mode]:
        g = g.to(device)
        ob = env.register(g, num_samples=eval_num_samples)
        batch_t = time()
        while True:
            with torch.no_grad():
                action = actor_critic.act(ob, g)

            ob, reward, done, info = env.step(action)
            
            if torch.all(done).item():
                cum_eval_sol += info['sol'].max(dim=1)[0].sum().cpu()
                batch_t = time() - batch_t
                T += batch_t

                n_rounds += 1
                cum_cnt += g.batch_size
                num_steps += max_efeat(g, env.t).sum().item()
                break

    actor_critic.train()
    avg_eval_sol = cum_eval_sol / cum_cnt
    T = T / n_rounds
    avg_steps = num_steps / cum_cnt
    return avg_eval_sol, T, avg_steps


if __name__ == "__main__":
    if args.mode == 'train':
        vali_results = []

        datasets = {"train": get_dataset(mode='train', **dataset_args), "vali": get_dataset(mode='vali', **dataset_args)}
        data_loaders = {
            "train":
            DataLoader(datasets["train"],
                       batch_size=rollout_batch_size,
                       shuffle=True,
                       collate_fn=collate_fn,
                       num_workers=0,
                       drop_last=True),
            "vali":
            DataLoader(datasets["vali"], batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        }

        rollout = RolloutStorage(max_t=max_rollout_t, batch_size=rollout_batch_size, num_samples=train_num_samples)
        framework = ProxPolicyOptimFramework(actor_critic=actor_critic,
                                             init_lr=init_lr,
                                             clip_value=clip_value,
                                             optim_num_samples=optim_num_samples,
                                             optim_batch_size=optim_batch_size,
                                             critic_loss_coef=critic_loss_coef,
                                             reg_coef=reg_coef,
                                             max_grad_norm=max_grad_norm,
                                             device=device)

        # Train Model
        for update_t in range(max_update_t):
            if update_t == 0 or torch.all(done).item():
                try:
                    g = next(train_data_iter)
                except:
                    train_data_iter = iter(data_loaders["train"])
                    g = next(train_data_iter)
                ob = env.register(g, num_samples=train_num_samples)
                rollout.insert_ob_and_g(ob, g)

            for step_t in range(max_rollout_t):
                with torch.no_grad():
                    (action, action_log_prob, value_pred) = actor_critic.act_and_crit(ob, g)

                ob, reward, done, info = env.step(action)

                rollout.insert_tensors(ob, action, action_log_prob, value_pred, reward, done)

                if torch.all(done).item():
                    avg_sol = info['sol'].max(dim=1)[0].mean().cpu()
                    break

            rollout.compute_rets_and_advantages(gamma)

            actor_loss, critic_loss, entropy_loss = framework.update(rollout)

            if (update_t + 1) % log_freq == 0:
                print("update_t: {:05d}".format(update_t + 1))
                print("train stats...")
                print("sol: {:.4f}, "
                      "actor_loss: {:.4f}, "
                      "critic_loss: {:.4f}, "
                      "entropy: {:.4f}".format(avg_sol, actor_loss.item(), critic_loss.item(), entropy_loss.item()))
                if (update_t + 1) % vali_freq == 0:
                    sol, T, steps = evaluate("vali", actor_critic)
                    print("vali stats...")
                    print("avg_sol: {:.4f}, avg_time: {:.4f}, avg_steps: {}".format(sol.item(), T, steps))
                    vali_results.append([sol.item(), T, steps])
                if (update_t + 1) % save_freq == 0:
                    # Save Model
                    idx = int((update_t + 1) / save_freq)
                    Save_Path = Model_Path / Path(Model_Name + "_{}.pt".format(idx))
                    torch.save(actor_critic.state_dict(), Save_Path)
                    print("Save Model at \"{}\"".format(Save_Path))

                    vali_df = pd.DataFrame(vali_results, columns=["sol", "ratio", "steps"])
                    vali_df.to_csv(Model_Path / Path(Model_Name + "_valis.csv"), mode='w', header=False)
                    print("Save Results")

    elif args.mode == 'test':
        datasets = {"test": get_dataset(mode='test', **dataset_args)}
        data_loaders = {
            "test": DataLoader(datasets["test"],
                               batch_size=eval_batch_size,
                               shuffle=False,
                               collate_fn=collate_fn,
                               num_workers=0)
        }
        # Load Model
        Save_Path = Model_Path / Path(Model_Name + "_{}.pt".format(1))
        if not Save_Path.exists():
            error_mess = "No Model at {}".format(Save_Path)
            raise ValueError(error_mess)
        state_dict = torch.load(Save_Path)
        actor_critic.load_state_dict(state_dict)
        print("Load Model from \"{}\"".format(Save_Path))
        # Test Model
        evaluate("test", actor_critic)
        sol, T, steps = evaluate("test", actor_critic)
        print("avg_sol: {}, avg_time: {}, avg_setps: {}".format(sol.item(), T, steps))
