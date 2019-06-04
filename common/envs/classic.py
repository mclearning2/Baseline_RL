from typing import Callable
from common.abstract.base_env import Gym

class Classic(Gym):
    def __init__(
        self,
        env_id: str,
        n_envs: int,
        render_available = False,
        max_episode: int = 50000,
        max_episode_steps: int = None,
        recent_score_len: int = 100,
        monitor_func: Callable = lambda x : x,
        clip_action: bool = True,
        scale_action: bool = False,
    ):
        super().__init__(
            env_id = env_id,
            n_envs = n_envs,
            render_available = render_available,
            max_episode = max_episode,
            max_episode_steps = max_episode_steps,
            recent_score_len = recent_score_len,
            monitor_func = monitor_func,
            clip_action = clip_action,
            scale_action = scale_action,
        )