import os
import sys
import gym
import hues
import wandb
import torch
import pickle
import random
import numpy as np
from glob import glob
from typing import Tuple, Dict, Union, Callable
from gym.wrappers import Monitor
from abc import ABC, abstractmethod

class BaseProject(ABC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def init_hyper_params(self) -> dict:
        '''You must return dictionary has hyperparameters for model
        
        Example:
            return { "gamma": 0.9531 }

        '''

    @abstractmethod
    def init_env(self) -> Tuple[gym.Env, Callable]:
        '''You must return environment with video_callable for recording

        Examples:
            env = gym.make('Pendulum-v0')
            video_callable = lambda x: x > 450 and x % 10 == 0
            return env, video_callable
        '''        

    @abstractmethod
    def init_models(self, env, hyper_params) \
            -> Dict[str, Union[torch.nn.Module, torch.optim.Optimizer]]:
        '''You must return models with optimizer
        
        Examples:
            model = MLP(...)
            optim = MLP(...)
            
        '''

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def __restore_wandb(self):
        
        paths = [ 
            self.config.models_path, 
            self.config.hyper_params_path
        ]
        user_name = self.config.user_name
        project = self.config.project
        run_id = self.config.run_id
        
        if run_id:

            run_path = os.path.join(user_name, project, run_id)

            hues.success(f"Loaded from {run_id}")

            for path in paths:
                root = os.path.dirname(path)
                name = os.path.basename(path)

                downloaded = wandb.restore(
                    name=name,
                    run_path=run_path,
                    replace=True,
                    root=root)

                txt = hues.huestr(f'{downloaded.name}').green.colorized
                hues.success(f"Saved to {txt}")

    def __init_hyper_params(self):
        path = self.config.hyper_params_path
        hyper_params = self.init_hyper_params()

        if self.config.restore:
            txt = hues.huestr(f'{path}').green.colorized
            try:
                with open(path, 'rb') as f:
                    unpickler = pickle.Unpickler(f)
                    hyper_params = unpickler.load()
                    hues.success("Loaded hyperparameters from " + txt)

            except (FileNotFoundError):
                hyper_params = self.init_hyper_params()
                hues.error("No such file or directory: " + txt)
                sys.exit() 
        else:
            hues.success("Hyperparameters are initialized")

        return hyper_params

    def __init_env(self):
        env, video_callable = self.init_env()
        if self.config.record:            
            env = Monitor(
                env=env,
                directory=self.config.video_dir,
                video_callable=video_callable,
                force=True)
            monitor_str = " with monitor"
        else:
            monitor_str = ""
         
        hues.success("Environment is initialized" + monitor_str)
            
        return env

    def __set_seed(self, env):
        env.seed(self.config.seed)
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        hues.info(f"Seeds are set by {self.config.seed}")

    def __init_models(self, env, hyper_params):
        models = self.init_models(env, hyper_params)

        if self.config.restore:
            txt = hues.huestr(f'{self.config.models_path}').green.colorized
            try:
                params = torch.load(self.config.models_path)
                for name in params.keys():
                    models[name].load_state_dict(params[name])

                hues.success("Loaded model and optimizer from " + txt)
            except FileNotFoundError:
                hues.error("No such file or directory: " + txt)
        else:
            hues.success("Models are initialized")

        return models

    def __init_agent(self, env, models, hyper_params):
        agent = self.init_agent(env, models, self.device, hyper_params)
        hues.success("Agent is initialized")

        return agent

    def __save_model(self, models):
        path = self.config.models_path

        dir_name = os.path.dirname(path)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        params = {name: tensor.state_dict() for name, tensor in models.items()}

        torch.save(params, path)
        wandb.save(path)

        txt = hues.huestr(f'{path}').green.colorized
        hues.success("Saved model and optimizer to " + txt)

    def __save_hyper_params(self, hyper_params):
        path = self.config.hyper_params_path

        dir_name = os.path.dirname(path)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        with open(path, 'wb+') as f:
            pickle.dump(hyper_params, f)
            wandb.save(path)

        txt = hues.huestr(f'{path}').green.colorized
        hues.success("Saved hyperparameters to " + txt)

    def __save_video(self):
        dir_name = self.config.video_dir

        files = glob(os.path.join(dir_name, "*.mp4"))
        for mp4_file in files:
            wandb.save(mp4_file)

        txt = hues.huestr(f'{dir_name}').green.colorized
        hues.success("Saved recorded videos to " + txt)

    def run(self):
        self.__restore_wandb()
        hyper_params = self.__init_hyper_params()
        env = self.__init_env()
        self.__set_seed(env)

        models = self.__init_models(env, hyper_params)
        agent = self.__init_agent(env, models, hyper_params)

        if self.config.test:
            self.test(agent, self.config.render)
        else:
            wandb.init(project=self.config.project)
            wandb.config.update(hyper_params)

            models_without_optims = list()
            for model in models.values():
                if isinstance(model, torch.nn.Module):
                    models_without_optims.append(model)

            wandb.watch(models_without_optims, log="parameters")

            try:
                self.train(agent, self.config.render)
            except KeyboardInterrupt:
                pass

            self.__save_model(models)
            self.__save_hyper_params(hyper_params)
            self.__save_video()
