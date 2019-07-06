# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environments and environment helper classes."""

import collections
import datetime
import os.path
import queue

import numpy as np
import tensorflow as tf
import imageio
import gym
from skimage.color import rgb2gray
from skimage.transform import resize

from utils import Vid_maker

nest = tf.contrib.framework.nest

COMPLETE_ACTION_SPACE = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN',
                         'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
                         'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
                         'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']

COMPLETE_ACTION_SET = tuple(range(len(COMPLETE_ACTION_SPACE)))


class LocaleCache:
    """Local level cache."""

    def __init__(self, cache_dir="/tmp/level_cache"):
        self._cache_dir = cache_dir
        tf.gfile.MakeDirs(cache_dir)

    def fetch(self, key, pk3_path):
        path = os.path.join(self._cache_dir, key)
        if tf.gfile.Exists(path):
            tf.gfile.Copy(path, pk3_path, overwrite=True)
            return True
        return False

    def write(self, key, pk3_path):
        path = os.path.join(self._cache_dir, key)
        if not tf.gfile.Exists(path):
            tf.gfile.Copy(pk3_path, path)


class PyProcessAtari:
    """Atari 2600 wrapper for PyProcess."""

    def __init__(
            self,
            game_name,
            config,
            num_action_repeats,
            seed):

        self._num_action_repeats = num_action_repeats
        self._random_state = np.random.RandomState(seed=seed)


        self._env = gym.make(game_name)
        self._atari_total_lives = self._env.unwrapped.ale.lives()
        self._show_display = config["show_display"]
        self._memory = queue.Queue(maxsize=config["n_input_frames"])

        self.action_map = self.get_action_map()

        # Saving video of game play
        self.save_video = config["save_video"]
        if self.save_video:
            height = self._env.observation_space.shape[0]
            width = self._env.observation_space.shape[1]
            self.video_obj = Vid_maker(game_name, height, width)



    def get_action_map(self):
        '''
        Returns the mapping between the action space of individual game and the complete action space.
        '''
        game_action_space = self._env.unwrapped.get_action_meanings()
        action_mapping = [game_action_space.index(i) if i in game_action_space else 0 for i in COMPLETE_ACTION_SPACE]
        return action_mapping

    def _reset(self):
        """
        Reset the environment using new random seed
        """
        self._env.seed(self._random_state.randint(0, 2 ** 31 - 1))

        initial_state = preprocess(self._env.reset())

        return initial_state

    def initial(self):
        """
        Initializes a new environment and refreshes the memory queue. The
        memory queue will only have multiple copies of same frame.
        """
        initial_state = self._reset()
        self._memory.queue.clear()
        while not self._memory.full():
            self._memory.put(initial_state)

        if self._show_display:
            self._env.render()
        mem_obs = np.stack(list(self._memory.queue), axis=2)
        return mem_obs

    def step(self, action_index):

        """
        Executes the given step in the environment
        """
        action_discrete = self.action_map[action_index]

        observation, reward, done, _ = self._env.step(action_discrete)
        if self.save_video:
            self.video_obj.add_frame(observation)
        observation = preprocess(observation)
        self._memory.get()

        self._memory.put(observation)
        mem_obs = np.stack(list(self._memory.queue), axis=2)
        if self._show_display:
            self._env.render()

        observation = mem_obs

        reward = np.array(reward, dtype=np.float32)

        if done:
            if self.save_video:
                self.video_obj.close()
            observation = self.initial()

        return reward, done, observation

    def close(self):
        """Closes the environment"""
        self._env.close()

    @staticmethod
    def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
        """Returns a nest of `TensorSpec` with the method's output specification."""
        width = constructor_kwargs["config"].get("width", 320)
        height = constructor_kwargs["config"].get("height", 240)
        n_input_frames = constructor_kwargs["config"].get("n_input_frames", 4)

        if method_name == "initial":
            return tf.contrib.framework.TensorSpec(
                [height, width, n_input_frames], tf.float64)
        elif method_name == "step":
            return (
                tf.contrib.framework.TensorSpec([], tf.float32),
                tf.contrib.framework.TensorSpec([], tf.bool),
                tf.contrib.framework.TensorSpec(
                    [height, width, n_input_frames], tf.float64)
            )


StepOutputInfo = collections.namedtuple(
    "StepOutputInfo", "episode_return episode_step")
StepOutput = collections.namedtuple(
    "StepOutput", "reward info done observation")


class FlowEnvironment:
    """An environment that returns a new state for every modifying method.

  The environment returns a new environment state for every modifying action and
  forces previous actions to be completed first. Similar to `flow` for
  `TensorArray`.
  """

    def __init__(self, env):
        """Initializes the environment.

    Args:
      env: An environment with `initial()` and `step(action)` methods where
        `initial` returns the initial observations and `step` takes an action
        and returns a tuple of (reward, done, observation). `observation`
        should be the observation after the step is taken. If `done` is
        True, the observation should be the first observation in the next
        episode.
    """
        self._env = env

    def initial(self):
        """Returns the initial output and initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. The reward and transition type in the `StepOutput` is the
      reward/transition type that lead to the observation in `StepOutput`.
    """
        with tf.name_scope("flow_environment_initial"):
            initial_reward = tf.constant(0.0)
            initial_info = StepOutputInfo(tf.constant(0.0), tf.constant(0))
            initial_done = tf.constant(True)
            initial_observation = self._env.initial()

            initial_output = StepOutput(
                initial_reward, initial_info, initial_done, initial_observation
            )

            # Control dependency to make sure the next step can't be taken before the
            # initial output has been read from the environment.
            with tf.control_dependencies(nest.flatten(initial_output)):
                initial_flow = tf.constant(0, dtype=tf.int64)
            initial_state = (initial_flow, initial_info)
            return initial_output, initial_state

    def step(self, action, state):
        """Takes a step in the environment.

    Args:
      action: An action tensor suitable for the underlying environment.
      state: The environment state from the last step or initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. On episode end (i.e. `done` is True), the returned reward
      should be included in the sum of rewards for the ending episode and not
      part of the next episode.
    """

        with tf.name_scope("flow_environment_step"):
            flow, info = nest.map_structure(tf.convert_to_tensor, state)

            # Make sure the previous step has been executed before running the next
            # step.
            with tf.control_dependencies([flow]):
                reward, done, observation = self._env.step(action)

            with tf.control_dependencies(nest.flatten(observation)):
                new_flow = tf.add(flow, 1)

            # When done, include the reward in the output info but not in the
            # state for the next step.
            new_info = StepOutputInfo(
                info.episode_return + reward, info.episode_step + 1
            )
            new_state = (
                new_flow,
                nest.map_structure(
                    lambda a, b: tf.where(done, a, b),
                    StepOutputInfo(tf.constant(0.0), tf.constant(0)),
                    new_info,
                ),
            )

            output = StepOutput(reward, new_info, done, observation)
            return output, new_state


def preprocess(vid_frame, out_dim=84):
    """
    Preprocess the given video frame by converting it into grayscale and
    resizing the image into the give output dimension.
    """
    assert vid_frame.ndim == 3, "3 channel image is required as observation"
    grayscale = rgb2gray(vid_frame)
    resized_frame = resize(
        grayscale, (out_dim, out_dim), mode="reflect", anti_aliasing=True
    )
    # resized_frame = resize(vid_frame, (out_dim, out_dim), mode='reflect', anti_aliasing=True)

    return resized_frame

# TODO: Implementation for ram version of environment
# TODO: Implement lives and test with breakout
