from tensorforce.agents import PPOAgent

from serpent.utilities import SerpentError

import numpy as np

# This file is borrowed from SerpentAIsaacGameAgentPlugin:
# https://github.com/SerpentAI/SerpentAIsaacGameAgentPlugin/blob/master/files/helpers/ppo.py
class SerpentPPO:

    def __init__(self, frame_shape=None, game_inputs=None):

        if frame_shape is None:
            raise SerpentError("A 'frame_shape' tuple kwarg is required...")

        states_spec = {"type": "float", "shape": frame_shape}

        if game_inputs is None:
            raise SerpentError("A 'game_inputs' dict kwarg is required...")

        self.game_inputs = game_inputs
        self.game_inputs_mapping = self._generate_game_inputs_mapping()

        actions_spec = {"type": "int", "num_actions": len(self.game_inputs)}

        summary_spec = {
            "directory": "./board/",
            "steps": 50,
            "labels": [
                "configuration",
                "gradients_scalar",
                "regularization",
                "inputs",
                "losses",
                "variables"
            ]
        }

        network_spec = [
            {"type": "conv2d", "size": 32, "window": 8, "stride": 4},
            {"type": "conv2d", "size": 64, "window": 4, "stride": 2},
            {"type": "conv2d", "size": 64, "window": 3, "stride": 1},
            {"type": "flatten"},
            {"type": "dense", "size": 1024}
        ]

        self.agent = PPOAgent(
            batched_observe=2560,
            scope="ppo",
            device=None,
            saver_spec=None,
            distributed_spec=None,
            discount=0.97,
            variable_noise=None,
            states_preprocessing_spec=None,
            explorations_spec=None,
            reward_preprocessing_spec=None,
            distributions_spec=None,
            entropy_regularization=0.01,
            batch_size=2560,
            keep_last_timestep=True,
            baseline_mode=None,
            baseline=None,
            baseline_optimizer=None,
            gae_lambda=None,
            likelihood_ratio_clipping=None,
            step_optimizer=None,
            optimization_steps=10
        )

    def generate_action(self, game_frame_buffer):
        states = np.stack(
            game_frame_buffer,
            axis=2
        )

        action = self.agent.act(states)
        label = self.game_inputs_mapping[action]

        return action, label, self.game_inputs[label]

    def observe(self, reward=0, terminal=False):
        self.agent.observe(reward=reward, terminal=terminal)

    def _generate_game_inputs_mapping(self):
        mapping = dict()

        for index, key in enumerate(self.game_inputs):
            mapping[index] = key

        return mapping