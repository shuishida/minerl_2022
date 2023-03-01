import argparse
import os
from collections import OrderedDict

import aicrowd_gym
import torch
from gym.spaces import Dict, Discrete
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


import sys
import torch.nn as nn

sys.path.append(".")


from src.minerl.dataset import EpisodeDataset
from const import get_env_name, VPT_PATH, ROOT_PATH
from src.minerl.agent import MineSORLAgent, load_base_agent

from src.utils.file_utils import load_yaml
from src.utils.gym_spaces_utils import transform_space
from src.utils.nested_utils import to_item, nested_rearrange


def setup_model(base_agent, config, model_name):
    action_space, _, action_inv_tf = transform_space(Dict(OrderedDict(camera=Discrete(121), buttons=Discrete(8641))))
    if model_name == "bc":
        from src.algo.bc.model import BCModel
        model_class = BCModel
    else:
        raise NotImplementedError
    model = model_class(base_agent, action_space, action_inv_tf, config["model"], config["agent"])
    return model


class RLSetup(LightningModule):
    def __init__(
            self,
            env_name: str,
            model_name: str,
            data_dir: str,
            model_path: str,
            weights_path: str,
            checkpoint: str,
            config: dict,
            verbose: bool = False,
            imitation: bool = True
    ):
        super(RLSetup, self).__init__()
        self.env_name = env_name
        self.verbose = verbose
        self.config = config
        self.data_dir = data_dir
        self.setup_config = config["setup"]
        self.imitation = imitation
        self.checkpoint = checkpoint

        # if os.path.exists(data_dir):
        #     if self.setup_config["clear"]:
        #         shutil.rmtree(data_dir)
        #         os.makedirs(data_dir)
        #     else:
        #         print(f"Loading existing data at {data_dir}")
        # else:
        #     os.makedirs(data_dir)

        self.env = aicrowd_gym.make(env_name)
        self.base_agent = load_base_agent(self.env, model_path, weights_path, DEVICE="cuda")
        self.agent = MineSORLAgent(self.base_agent, self._sample_action)
        self.model = setup_model(self.base_agent, config, model_name)
        self.action_space, self.action_tf, self.action_inv_tf = transform_space(Dict(OrderedDict(camera=Discrete(121), buttons=Discrete(8641))))
        self.dataset = EpisodeDataset(data_dir, self.action_space, lambda a: self.action_tf(self.base_agent._env_action_to_agent(a)), config["buffer"]["sample_length"])
        # self.buffer = EpisodeBuffer(self.env.observation_space, self.env.action_space, self.config["agent"].get("n_options"), data_dir, **config["buffer"])
        self.batch_size = self.setup_config["batch_size"]
        self.save_hyperparameters()

    def _sample_action(self, state) -> MineSORLAgent:
        return self.model.sample_action(state)

    # def on_fit_start(self):
    #     n_steps = self.setup_config["prefill"] - len(self.buffer)
    #     agent = RandomAgent(self.buffer, self.env.observation_space, self.env.action_space)
    #     if n_steps:
    #         agent.rollouts(self.env, n_steps=n_steps, is_train=True)
    #
    # def on_train_epoch_start(self):
    #     stats = self.agent.rollouts(self.env, n_steps=self.setup_config["steps_per_epoch"], is_train=True)
    #     sys.stdout.write("\033[K")
    #     self.log_dict({"returns": stats["avg_returns"], "steps": float(stats["count_steps"]), "episodes": float(stats["count_episodes"])},
    #                   prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        output = self.model(*batch)
        if isinstance(output, tuple):
            loss, info = output
        else:
            loss = output
            info = {}
        self.log("loss", loss)
        if self.verbose:
            print(to_item(info))
        self.log_dict(info, prog_bar=True, logger=True)
        return loss

    def train_dataloader(self):
        # dataset = IterEpisodeDataset(self.buffer,
        #                              max_steps=self.setup_config["steps_per_epoch"] * self.setup_config["train_rollout_ratio"],
        #                              obsv_transforms=self.obsv_tf, action_transforms=self.action_tf)

        def collate_fn(batch):
            batch = default_collate(batch)
            return nested_rearrange(batch, "b l ... -> l b ...")

        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.setup_config["n_workers"], collate_fn=collate_fn,
                          shuffle=self.imitation)

    def on_after_backward(self):
        if self.global_step % 1000 == 0:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint))
            #         for name, param in self.model.named_parameters():
            #             self.logger.experiment.add_histogram(name, param, global_step)
            #             if param.requires_grad and param.grad is not None:
            #                 self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    def on_train_epoch_end(self):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint))
        torch.save(self.model.state_dict(), os.path.join(self.logger.log_dir, "checkpoints", f"{self.env_name}.pt"))
    #     print()
    #     if self.current_epoch % 10 == 0:
    #         self.evaluation()

    # def on_train_end(self):
        # print()
        # self.evaluation()

    # def evaluation(self):
    #     print("=== Running evaluation ===")
    #     save_dir = os.path.join(self.data_dir, f"epoch_{self.current_epoch}")
    #     buffer = EpisodeBuffer(self.env.observation_space, self.env.action_space, self.config["agent"].get("n_options"), save_dir,
    #                            **self.config["buffer"], clear=True)
    #     self.eval_env.reset()
    #     stats = self.agent.rollouts(self.eval_env, buffer, n_episodes=self.setup_config["n_eval_episodes"],
    #                                 is_train=False, save_pred=True, record_dir=save_dir)
    #     sys.stdout.write("\033[K")
    #     print(f"Eval: average returns: {stats['avg_returns']}")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.setup_config["lr"], weight_decay=0.0)

    @classmethod
    def run(cls, args: dict = None, parser: argparse.ArgumentParser = None):
        if not args:
            if not parser:
                parser = argparse.ArgumentParser(description='Generic RL setup')
            parser.add_argument('--env', help='env name', default='cave')
            parser.add_argument('--model', help='model name', default='dvon')
            parser.add_argument('--verbose', '-v', action="store_true", default=False)
            parser.add_argument('--devices', '-d', nargs='+', type=int, default=(0,))
            parser.add_argument('--epochs', '-e', type=int, default=100)

            args = vars(parser.parse_args())
        config = load_yaml(os.path.join(ROOT_PATH, f"./config/{args['model']}.yaml"))
        algo = config["agent"]["name"]
        env_name = get_env_name(args["env"])
        imitation = config["agent"].get("imitation", False)

        # For reproducibility
        manual_seed = config['setup'].get('manual_seed')
        if manual_seed:
            seed_everything(manual_seed, True)

        logger = TensorBoardLogger(save_dir=config['setup']['log_dir'], name=f"{algo}/{env_name}")
        data_dir = os.path.join(config['setup']['data_dir'], env_name) if imitation else os.path.join(config['setup']['data_dir'], algo, env_name)
        model_path = os.path.join(VPT_PATH, config["agent"]["model"])
        weights_path = os.path.join(VPT_PATH, config["agent"]["weights"])
        checkpoint = os.path.join("train", algo, f"{env_name}.pt")
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)

        setup = cls(env_name=env_name, model_name=args["model"], data_dir=data_dir, model_path=model_path, weights_path=weights_path, checkpoint=checkpoint,
                    config=config, verbose=args.get("verbose", False), imitation=imitation)

        trainer = Trainer(logger=logger,
                          callbacks=[
                              LearningRateMonitor(),
                              ModelCheckpoint(save_top_k=2,
                                              dirpath=os.path.join(logger.log_dir, "checkpoints"),
                                              monitor="loss",
                                              save_last=True),
                          ],
                          devices=args.get("devices", (0,)),
                          max_epochs=args.get("epochs", 100),
                          accelerator="gpu")

        trainer.fit(setup)
        return setup.agent, setup.model, setup, trainer


if __name__ == "__main__":
    RLSetup.run()
