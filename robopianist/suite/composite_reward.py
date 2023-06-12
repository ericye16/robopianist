# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility class for composite reward functions."""

from typing import Callable, Dict

from dm_control import mjcf

Reward = float
RewardFn = Callable[[mjcf.Physics], Reward]


class CompositeReward:
    """A reward function composed of individual reward terms.

    Useful for grouping sub-rewards of a task into a single reward function, computing
    their sum, and logging the individual terms.
    """

    def __init__(self, **kwargs) -> None:
        self._reward_fns: Dict[str, RewardFn] = {}
        for name, reward_fn in kwargs.items():
            self.add(name, reward_fn)
        self._reward_terms: Dict[str, Reward] = {}

    def add(self, name: str, reward_fn: RewardFn) -> None:
        """Adds a reward term to the reward terms."""
        self._reward_fns[name] = reward_fn

    def remove(self, name: str) -> None:
        """Removes a reward term from the reward terms."""
        del self._reward_fns[name]

    def compute(self, physics: mjcf.Physics) -> float:
        """Computes the reward terms sequentially and returns their sum.

        Note that the reward terms are computed in the order they were added.
        """
        sum_of_rewards = 0.0
        for name, reward_fn in self._reward_fns.items():
            rew = reward_fn(physics)
            sum_of_rewards += rew
            self._reward_terms[name] = rew
        return sum_of_rewards

    @property
    def reward_fns(self) -> Dict[str, RewardFn]:
        return self._reward_fns

    @property
    def reward_terms(self) -> Dict[str, Reward]:
        return self._reward_terms



class TieredReward:
    """A reward function composed of individual reward terms.

    Useful for grouping sub-rewards of a task into a single reward function, computing
    their sum, and logging the individual terms.
    """

    def __init__(self, key_press_reward, sustain_reward, energy_reward) -> None:
        self.key_press_reward_ = key_press_reward
        self.sustain_reward_ = sustain_reward
        self.energy_reward_ = energy_reward
        self.fingering_reward_ = None
        self.forearm_reward_ = None

    def add(self, name: str, reward_fn: RewardFn) -> None:
        """Adds a reward term to the reward terms."""
        if name == "fingering_reward":
            self.fingering_reward_ = reward_fn
        elif name == "forearm_reward":
            self.forearm_reward_ = reward_fn
        else:
            raise ValueError(f"Cannot add {name}")

    def remove(self, name: str) -> None:
        """Removes a reward term from the reward terms."""
        raise ValueError(f"cannot remove {name}")

    def compute(self, physics: mjcf.Physics) -> float:
        """Computes the reward terms sequentially and returns their sum.

        Note that the reward terms are computed in the order they were added.
        """
        key_press_rew = self.key_press_reward_(physics)
        sustain_reward = self.sustain_reward_(physics)
        energy_reward = self.energy_reward_(physics)
        fingering_reward =  self.fingering_reward_(physics) if self.fingering_reward_ else 0.0
        if not self.fingering_reward_:
            assert False
        forearm_reward = self.forearm_reward_(physics) if self.forearm_reward_ else 0.0
        sum_of_rewards = key_press_rew + sustain_reward + fingering_reward
        if key_press_rew > 0.5:
            sum_of_rewards += energy_reward + forearm_reward

        return sum_of_rewards

