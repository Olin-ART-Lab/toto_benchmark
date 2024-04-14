"""Train a TOTO agent.

Example command:
python train.py --config-name train_bc.yaml

Hyperparameters can be set in corresponding .yaml files in confs/
"""

import logging
import os
import pickle

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader, random_split
import wandb

import baselines
import toto_benchmark
from toto_benchmark.agents import init_agent_from_config
from toto_benchmark.vision import EMBEDDING_DIMS
from dataset_traj import FrankaDatasetTraj
from toto_benchmark.sim.eval_agent import eval_agent, create_agent_predict_fn

log = logging.getLogger(__name__)

def global_seeding(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

config_path = os.path.join(os.path.dirname(toto_benchmark.__file__), 'conf')
@hydra.main(config_path=config_path, config_name="train_bc")
def main(cfg : DictConfig) -> None:
    with open_dict(cfg):
        cfg['saved_folder'] = os.getcwd()
        print("Model saved dir: ", cfg['saved_folder'])

        if 'crop' not in cfg['data']['images']:
            cfg['data']['images']['crop'] = False
        if 'H' not in cfg['data']:
            cfg['data']['H'] = 1
        #cfg['data']['logs_folder'] = os.path.dirname(cfg['data']['pickle_fn'])

    if cfg.agent.type in ['bcimage', 'bcimage_pre']:
        cfg['data']['images']['per_img_out'] = EMBEDDING_DIMS[cfg['agent']['vision_model']]
        if cfg.agent.type == 'collab_agent':
            # assume in_dim is without adding the image embedding dimensions
            cfg['data']['in_dim'] = cfg['data']['in_dim'] + cfg['data']['images']['per_img_out']

    print(OmegaConf.to_yaml(cfg, resolve=True))

    with open(os.path.join(os.getcwd(), 'hydra.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    global_seeding(cfg.training.seed)
    #print(hydra.utils.get_original_cwd(),cfg.data.pickle_fn)

    agent_name = cfg['saved_folder'].split('outputs/')[-1]
    flat_dict = {}
    for key in ['data', 'agent', 'training']:
        flat_dict.update(dict(cfg[key]))
    wandb.init(project="toto-bc", config=flat_dict)
    wandb.run.name = "{}".format(agent_name)

    '''
    # modify the default parameters of np.load
        np_load_old = np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        all_data = []
        dir_of_trajs = "/home/jess/sweep_data"
        for root, dirs, files in os.walk(dir_of_trajs):
            #breakpoint()
            print(files)
            if os.path.basename(root) == "train" or os.path.basename(root) == "val":
                for filename in files:
                    #print(filename)
                    file_path = os.path.join(root, filename)
                    #breakpoint()
                    data_traj = np.load(file_path)
                    all_data.extend(data_traj)
        all_data = all_data[0:40]
        print(len(all_data))
        breakpoint()
    '''

    #breakpoint()
    
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    #all_data = np.load("/home/jess/toto_p2/toto_benchmark/toto_benchmark/embedded_data.npy")
    

    #dset_pnp_push_sweep = FrankaDatasetTraj(convert_to_arrays(all_data), cfg, sim=cfg.data.sim)
    #breakpoint()
    #dset_sweep = FrankaDatasetTraj(convert_to_arrays(all_data), cfg, sim=cfg.data.sim)
    #np.save("/home/jess/toto_p2/toto_benchmark/toto_benchmark/sweep.npy", dset_sweep)
    #np.save("/home/jess/toto_p2/toto_benchmark/toto_benchmark/pnp_push_sweep.npy", dset_pnp_push_sweep)
    #del all_data
    #breakpoint()
    #del all_data2
    #breakpoint()
    dset_sweep = np.load("/home/jess/toto_p2/toto_benchmark/toto_benchmark/sweep.npy") 
    split_sizes = [int(len(dset_sweep) * 0.8), len(dset_sweep) - int(len(dset_sweep) * 0.8)]
    train_set, test_set = random_split(dset_sweep, split_sizes)

    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=cfg.training.batch_size, \
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.training.batch_size)
    agent, _ = init_agent_from_config(cfg, cfg.training.device, normalization=dset_sweep) 
    train_metric, test_metric = baselines.Metric(), baselines.Metric()

    for epoch in range(cfg.training.epochs):
        acc_loss = 0.
        train_metric.reset()
        test_metric.reset()
        batch = 0
        for data in train_loader:
            #breakpoint()
            #print("training")
            for key in data:
                #print(f"key: {key}")
                #print(f"data key: {data[key]}")
                data[key] = data[key].to(cfg.training.device)
            agent.train(data)
            acc_loss += agent.loss
            train_metric.add(agent.loss.item())
            print('epoch {} \t batch {} \t train {:.6f}'.format(epoch, batch, agent.loss.item()), end='\r')
            batch += 1

        for data in test_loader:
            #print("testing")
            for key in data:
                data[key] = data[key].to(cfg.training.device)
            test_metric.add(agent.eval(data))

        log.info('epoch {} \t train {:.6f} \t test {:.6f}'.format(epoch, train_metric.mean, test_metric.mean))
        print(f"epoch: {epoch}, train metric mean: {train_metric.mean}, test metric mean: {test_metric.mean}")
        log.info(f'Accumulated loss: {acc_loss}')
        if epoch % cfg.training.save_every_x_epoch == 0:
            agent.save(os.getcwd())

        wandb.log({"Train Loss": train_metric.mean, "Epoch": epoch})
        wandb.log({"Test Loss": test_metric.mean, "Epoch": epoch})
        wandb.log({"Acc Train Loss": acc_loss, "Epoch": epoch})

    agent.save(os.getcwd())

    if cfg.data.sim:
        # Evaluate the simulation agent online
        eval_agent(create_agent_predict_fn(agent, cfg))

    log.info("Saved agent to {}".format(os.getcwd()))

    
def convert_to_arrays(obj):
    if isinstance(obj, list):  # If the current object is a list, convert it to a NumPy array
        return np.array([convert_to_arrays(item) for item in obj])
    elif isinstance(obj, dict):  # If it's a dictionary, apply this function to each of its values
        return {key: convert_to_arrays(value) for key, value in obj.items()}
    else:  # If it's neither (meaning it's a leaf node in the structure), return it as is
        return obj

if __name__ == '__main__':
    main()
