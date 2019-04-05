import os
import sys
import gym
import wandb
import torch
import pickle
import random
import numpy as np
from glob import glob
from typing import Tuple, Dict, Union, Callable
from gym.wrappers import Monitor
from abc import ABC, abstractmethod
from common.utils.path import check_or_make_dir

class BaseProject(ABC):
    ''' 구현한 알고리즘이 좀 더 쉽게 원하는 환경에 적용될 수 있게 지원하는 프로젝트
        기본적으로 구현에 있어 필요한 것들
    
    Functions to need to implement :
        init_hyper_params() -> dict
            : logging을 위해 기록해야할 hyperparameter
        init_env: 모델이 학습할 OpenAI atari gym. 
        init_models: agent
    '''
    def __init__(self, config):

        self.user_name = config.user_name
        self.project = config.project
        self.run_id = config.run_id
        
        self.seed = config.seed
        
        self.test_mode = config.test_mode
        self.restore = config.restore
        self.render = config.render
        self.record = config.record

        self.report_dir = config.report_dir

        self.video_dir = config.video_dir
        self.params_path = config.params_path
        self.hyperparams_path = config.hyperparams_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def init_hyper_params(self) -> dict:
        ''' hyperparameter를 dictionary에 담고 반환

            config.restore가 true인 경우 이 변수 대신 report/{project}/hyperparams.pkl
            에서 불러온 변수를 사용한다.
        
        Example:
            return { "gamma": 0.9531 }

        '''

    @abstractmethod
    def init_env(self, hyper_params) -> Tuple[gym.Env]:
        ''' 환경을 구현 후 반환
            
            이후 env.seed(..)형태로 seed 결정 및
            action_size 크기 결정. TODO: Atari같은 픽셀의 경우도 고려

        Examples:
            env = gym.make('Pendulum-v0')
            return env
        '''        

    @abstractmethod
    def init_models(self, input_size, output_size, hyper_params) \
            -> Dict[str, Union[torch.nn.Module, torch.optim.Optimizer]]:
        ''' Agent가 사용할 모델을 구성후 반환
        
        Examples:
            model = MLP(...)
            optimizer = optim(...)

            return {'model': model, 'optim': optimizer} 
        '''

    @abstractmethod
    def init_agent(self, env, models, device, hyper_params):
        '''You must return agent

        Examples:
            return A2C(...)

        '''

    @abstractmethod
    def train(self, agent,):
        pass

    @abstractmethod
    def test(self, agnet, render):
        pass

    def monitor_func(self, video_callable=None, *args, **kargs):
        def func(env):
            if self.record:
                return Monitor(
                    env=env,
                    directory=self.video_dir,
                    video_callable=video_callable,
                    force=True,
                    *args,
                    **kargs
                )
            
            else:
                return env

        return func

    def _restore_wandb(self):
        ''' wandb로부터 학습에 필요한 model과 hyperparameter를 복원
            복원하는 경로
             - config.models_path
             - config.hyper_params_path
        '''

        if self.run_id:
            run_path = os.path.join(self.user_name, self.project, self.run_id)

            print(f"[INFO] Loaded from {run_path}")
            #TODO: root와 name을 나누지 않아도 되는지
            for path in [self.params_path, self.hyperparams_path]:
                root = os.path.dirname(path)
                name = os.path.basename(path)

                downloaded = wandb.restore(
                    name=name,
                    run_path=run_path,
                    replace=True,
                    root=root)

                print(f"Saved to {path}")

    def _init_hyper_params(self):
        ''' pickle 형식(.pkl)으로 저장했던 hyperparameter를 다시 불러온다.
            config.restore 일 때만 반환하며 아닐 경우 None을 반환.
            만약 파일이 없을 경우 에러를 보이며 반환
        '''

        #TODO 에러가 어떻게 나는지 확인
        if self.restore:
            with open(self.hyper_params_path, 'rb') as f:
                unpickler = pickle.Unpickler(f)
                hyper_params = unpickler.load()

                print("[INFO] Loaded hyperparameters from " + self.hyper_params_path)

            return hyper_params
        else:
            print("[INFO] Initialized hyperparameters")
            return self.init_hyper_params()

    def _set_seed(self, env):
        env.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        print(f"[INFO] Seeds are set by {self.seed}")

    def _init_models(self, env, hyper_params):
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_size = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            action_size = env.action_space.shape[0]

        models = self.init_models(
                    input_size=env.observation_space.shape[0], 
                    output_size=action_size,
                    hyper_params=hyper_params
                )

        if self.restore:
            #TODO: 파일이 없을 경우 에러 구하기
            params = torch.load(self.params_path)
            for name in params.keys():
                models[name].load_state_dict(params[name])

            print("[INFO] Loaded model and optimizer from " + self.params_path)
        else:
            print("[INFO] Initialized models ")
        
        return models

    def _save_model(self, models):
        check_or_make_dir(self.params_path)

        params = {name: tensor.state_dict() for name, tensor in models.items()}

        torch.save(params, self.params_path)
        wandb.save(self.params_path)

        print("[INFO] Saved model and optimizer to " + self.params_path)

    def _save_hyper_params(self, hyper_params):
        check_or_make_dir(self.hyperparams_path)

        with open(self.hyperparams_path, 'wb+') as f:
            pickle.dump(hyper_params, f)
            wandb.save(self.hyperparams_path)

        print("[INFO] Saved hyperparameters to " + self.hyperparams_path)

    def _save_video(self):
        files = glob(os.path.join(self.video_dir, "*.mp4"))
        for mp4_file in files:
            wandb.save(mp4_file)

        print("[INFO] Saved recorded videos to " + self.video_dir)

    def run(self):
        self._restore_wandb()
        
        hyper_params = self._init_hyper_params()
        env = self.init_env(hyper_params)

        self._set_seed(env)

        models = self._init_models(env, hyper_params)
        agent = self.init_agent(env, models, self.device, hyper_params)
        print("[INFO] Initialized agent")

        if self.test_mode:
            print("[INFO] Starting test...")
            self.test(agent, self.render)
        else:
            print("[INFO] Starting train...")
            wandb.init(project=self.project, config=hyper_params)

            params = [model for model in models.values() \
                            if isinstance(model, torch.nn.Module)]

            wandb.watch(params, log="parameters")

            try:
                self.train(agent)
            except KeyboardInterrupt:
                pass

            self._save_model(models)
            self._save_hyper_params(hyper_params)
            self._save_video()
        
        env.close()
