import time
import os
import pickle
import serpent.cv

import numpy as np
import collections

from datetime import datetime

from serpent.frame_transformer import FrameTransformer
from serpent.frame_grabber import FrameGrabber
from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey, KeyboardEvent, KeyboardEvents
from serpent.input_controller import MouseButton, MouseEvent, MouseEvents

from .helpers.game_status import Game
from .helpers.terminal_printer import TerminalPrinter
from .helpers.ppo import SerpentPPO
from serpent.enums import InputControlTypes


class SerpentFortGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.printer = TerminalPrinter()

    def setup_play(self):
        self.game_inputs = {
            "MOVE UP": [
                KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W)
            ],
            "MOVE LEFT": [
                KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_A)
            ],
            "MOVE DOWN": [
                KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_S)
            ],
            "MOVE RIGHT": [
                KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_D)
            ],
            "MOVE TOP-LEFT": [
                KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W),
                KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_A)
            ],
            "MOVE TOP-RIGHT": [
                KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_W),
                KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_D)
            ],
            "MOVE DOWN-LEFT": [
                KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_S),
                KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_A)
            ],
            "MOVE DOWN-RIGHT": [
                KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_S),
                KeyboardEvent(KeyboardEvents.DOWN, KeyboardKey.KEY_D)
            ],
            "SHOOT UP": [
                MouseEvent(MouseEvents.CLICK, MouseButton.LEFT)
            ],
            "SHOOT LEFT": [
                MouseEvent(MouseEvents.CLICK, MouseButton.RIGHT)
            ],
            "DON'T MOVE": []
        }

        self.ppo_agent = SerpentPPO(
            frame_shape=(125, 112, 4),
            game_inputs=self.game_inputs
        )

        self.first_run = True
        self.game_over = False
        self.run_count = 0
        self.run_reward = 0

        self.observation_count = 0
        self.episode_observation_count = 0

        self.performed_inputs = collections.deque(list(), maxlen=8)

        self.reward_10 = collections.deque(list(), maxlen=10)
        self.reward_100 = collections.deque(list(), maxlen=100)
        self.reward_1000 = collections.deque(list(), maxlen=1000)

        self.rewards = list()

        self.average_reward_10 = 0
        self.average_reward_100 = 0
        self.average_reward_1000 = 0

        self.top_reward = 0
        self.top_reward_run = 0

        self.previous_score = 0

        self.score_10 = collections.deque(list(), maxlen=10)
        self.score_100 = collections.deque(list(), maxlen=100)
        self.score_1000 = collections.deque(list(), maxlen=1000)

        self.average_score_10 = 0
        self.average_score_100 = 0
        self.average_score_1000 = 0

        self.best_score = 0
        self.best_score_run = 0

        self.just_relaunched = False

        self.frame_buffer = None

        try:
            self.ppo_agent.agent.restore_model(
                directory=os.path.join(
                    os.getcwd(), "datasets", "pacai"))
            self.restore_metadata()
        except Exception:
            pass

        self.analytics_client.track(
            event_key="INITIALIZE", data=dict(
                episode_rewards=[]))

        for reward in self.rewards:
            self.analytics_client.track(
                event_key="EPISODE_REWARD", data=dict(
                    reward=reward))
            time.sleep(0.01)

        # Warm Agent?
        game_frame_buffer = FrameGrabber.get_frames(
            [0, 1, 2, 3], frame_type="PIPELINE")
        game_frame_buffer = self.extract_game_area(game_frame_buffer)
        self.ppo_agent.generate_action(game_frame_buffer)

        self.score = collections.deque(np.full((16,), 0), maxlen=16)
        self.lives = collections.deque(np.full((16,), 3), maxlen=16)
        self.continuity_bonus = 0

        self.started_at = datetime.utcnow().isoformat()
        self.episode_started_at = None

        self.paused_at = None

        print("Enter - Auto Save")
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(2)
        print("Enter - Menu")
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(1)
        print("Enter - Start game")
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(1)

        # Make sure to initialize Game() after passing the Start game menu,
        # otherwise the pointers may not be fully loaded.
        self.game_data = Game()
        return

    def handle_play(self, game_frame):
        if self.first_run:
            self.run_count += 1
            self.first_run = False
            self.episode_started_at = time.time()

            return None

        self.printer.add("")
        self.printer.add("Log234 - Pac-AI")
        self.printer.add("Reinforcement Learning: Training a PPO Agent")
        self.printer.add("")
        self.printer.add(f"Stage Started At: {self.started_at}")
        self.printer.add(f"Current Run: #{self.run_count}")
        self.printer.add("")

        if self.game_data.IsPaused():
            if self.paused_at is None:
                self.paused_at = time.time()

            # Give ourselves 30 seconds to work with
            if time.time() - self.paused_at >= 30:
                self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
                time.sleep(1)
                return

            self.printer.add("The game is paused.")
            self.printer.flush()
            return
        else:
            self.paused_at = None

        self.score.appendleft(self.game_data.GetScore())
        self.printer.add(f"Score: {self.score[0]}")
        self.lives.appendleft(self.game_data.GetLives())
        self.printer.add(f"Lives: {self.lives[0]}")

        reward = self.reward_agent()

        self.printer.add(f"Current Reward: {round(reward, 2)}")
        self.printer.add(f"Run Reward: {round(self.run_reward, 2)}")
        self.printer.add("")

        if self.frame_buffer is not None:
            self.run_reward += reward

            self.observation_count += 1
            self.episode_observation_count += 1

            self.analytics_client.track(
                event_key="RUN_REWARD", data=dict(
                    reward=reward))

            if self.ppo_agent.agent.batch_count == self.ppo_agent.agent.batch_size - 1:
                self.printer.flush()
                self.printer.add("")
                self.printer.add("Updating Pac-AI Model With New Data... ")
                self.printer.flush()

                self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
                self.ppo_agent.observe(
                    reward, terminal=(
                        self.game_data.IsOver()))
                self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)

                self.frame_buffer = None

                if not self.game_data.IsOver():
                    time.sleep(1)
                    return None
            else:
                self.ppo_agent.observe(
                    reward, terminal=(
                        self.game_data.IsOver()))

        self.printer.add(f"Observation Count: {self.observation_count}")
        self.printer.add(
            f"Episode Observation Count: {self.episode_observation_count}")
        self.printer.add(
            f"Current Batch Size: {self.ppo_agent.agent.batch_count}")
        self.printer.add("")

        if not self.game_data.IsOver():
            self.death_check = False

            self.printer.add(
                f"Continuity Bonus: {round(self.continuity_bonus, 2)}")
            self.printer.add("")
            self.printer.add(
                f"Average Rewards (Last 10 Runs): {round(self.average_reward_10, 2)}")
            self.printer.add(
                f"Average Rewards (Last 100 Runs): {round(self.average_reward_100, 2)}")
            self.printer.add(
                f"Average Rewards (Last 1000 Runs): {round(self.average_reward_1000, 2)}")
            self.printer.add("")
            self.printer.add(
                f"Top Run Reward: {round(self.top_reward, 2)} (Run #{self.top_reward_run})")
            self.printer.add("")
            self.printer.add(
                f"Previous Run Score: {round(self.previous_score, 2)}")
            self.printer.add("")
            self.printer.add(
                f"Average Score (Last 10 Runs): {round(self.average_score_10, 2)}")
            self.printer.add(
                f"Average Score (Last 100 Runs): {round(self.average_score_100, 2)}")
            self.printer.add(
                f"Average Score (Last 1000 Runs): {round(self.average_score_1000, 2)}")
            self.printer.add("")
            self.printer.add(
                f"Best Score: {round(self.best_score, 2)} (Run #{self.best_score_run})")
            self.printer.add("")
            self.printer.add("Latest Inputs:")
            self.printer.add("")

            for i in self.performed_inputs:
                self.printer.add(i)

            self.printer.flush()

            self.frame_buffer = FrameGrabber.get_frames(
                [0, 1, 2, 3], frame_type="PIPELINE")
            self.frame_buffer = self.extract_game_area(self.frame_buffer)

            action, label, game_input = self.ppo_agent.generate_action(
                self.frame_buffer)

            self.performed_inputs.appendleft(label)
            self.input_controller.handle_keys(game_input)
        else:
            self.input_controller.handle_keys([])
            self.analytics_client.track(
                event_key="RUN_END", data=dict(
                    run=self.run_count))

            self.printer.add("Game Over.")
            self.printer.flush()
            self.run_count += 1

            self.reward_10.appendleft(self.run_reward)
            self.reward_100.appendleft(self.run_reward)
            self.reward_1000.appendleft(self.run_reward)

            self.rewards.append(self.run_reward)

            self.average_reward_10 = float(np.mean(self.reward_10))
            self.average_reward_100 = float(np.mean(self.reward_100))
            self.average_reward_1000 = float(np.mean(self.reward_1000))

            if self.run_reward > self.top_reward:
                self.top_reward = self.run_reward
                self.top_reward_run = self.run_count - 1

                self.analytics_client.track(
                    event_key="NEW_RECORD",
                    data=dict(
                        type="REWARD",
                        value=self.run_reward,
                        run=self.run_count - 1))

            self.analytics_client.track(
                event_key="EPISODE_REWARD", data=dict(
                    reward=self.run_reward))

            self.previous_score = max(list(self.score)[:4])

            self.run_reward = 0

            self.score_10.appendleft(self.previous_score)
            self.score_100.appendleft(self.previous_score)
            self.score_1000.appendleft(self.previous_score)

            self.average_score_10 = float(np.mean(self.score_10))
            self.average_score_100 = float(np.mean(self.score_100))
            self.average_score_1000 = float(np.mean(self.score_1000))

            if self.previous_score > self.best_score:
                self.best_score = self.previous_score
                self.best_score_run = self.run_count - 1

                self.analytics_client.track(
                    event_key="NEW_RECORD",
                    data=dict(
                        type="score",
                        value=self.previous_score,
                        run=self.run_count - 1))

            if not self.run_count % 10:
                self.ppo_agent.agent.save_model(
                    directory=os.path.join(
                        os.getcwd(),
                        "datasets",
                        "pacai",
                        "ppo_model"),
                    append_timestep=False)
                self.dump_metadata()

            self.lives = collections.deque(np.full((16,), 3), maxlen=16)
            self.score = collections.deque(np.full((16,), 0), maxlen=16)

            self.multiplier_damage = 0

            self.performed_inputs.clear()

            self.frame_buffer = None

            self.input_controller.tap_key(KeyboardKey.KEY_ENTER, duration=1.5)

            self.episode_started_at = time.time()
            self.episode_observation_count = 0

    def reward_agent(self):
        if self.game_data.IsOver() or self.lives[0] < self.lives[1]:
            return 0

        if (self.score[0] > self.score[1]):
            if self.best_score is not 0:
                reward = self.score[0] / self.best_score
            else:
                reward = self.score[0] / 700
            if reward > 1:
                reward = 1

        else:
            reward = 0

        return reward

    def reward_agent_backup(self):
        if self.game_data.IsOver():
            return 0

        if self.lives[0] < self.lives[1]:
            self.continuity_bonus = 0
            return 0
        elif self.score[0] > self.score[1]:
            self.continuity_bonus += 0.05

            if self.continuity_bonus > 1:
                self.continuity_bonus = 1

            return (1 * self.continuity_bonus) + 0.001
        else:
            return 0.001

    def dump_metadata(self):
        metadata = dict(
            started_at=self.started_at,
            run_count=self.run_count - 1,
            observation_count=self.observation_count,
            reward_10=self.reward_10,
            reward_100=self.reward_100,
            reward_1000=self.reward_1000,
            rewards=self.rewards,
            average_reward_10=self.average_reward_10,
            average_reward_100=self.average_reward_100,
            average_reward_1000=self.average_reward_1000,
            top_reward=self.top_reward,
            top_reward_run=self.top_reward_run,
            score_10=self.score_10,
            score_100=self.score_100,
            score_1000=self.score_1000,
            average_score_10=self.average_score_10,
            average_score_100=self.average_score_100,
            average_score_1000=self.average_score_1000,
            best_score=self.best_score,
            best_score_run=self.best_score_run
        )

        with open("datasets/pacai/metadata.json", "wb") as f:
            f.write(pickle.dumps(metadata))

    def restore_metadata(self):
        with open("datasets/pacai/metadata.json", "rb") as f:
            metadata = pickle.loads(f.read())

        self.started_at = metadata["started_at"]
        self.run_count = metadata["run_count"]
        self.observation_count = metadata["observation_count"]
        self.reward_10 = metadata["reward_10"]
        self.reward_100 = metadata["reward_100"]
        self.reward_1000 = metadata["reward_1000"]
        self.rewards = metadata["rewards"]
        self.average_reward_10 = metadata["average_reward_10"]
        self.average_reward_100 = metadata["average_reward_100"]
        self.average_reward_1000 = metadata["average_reward_1000"]
        self.top_reward = metadata["top_reward"]
        self.top_reward_run = metadata["top_reward_run"]
        self.score_10 = metadata["score_10"]
        self.score_100 = metadata["score_100"]
        self.score_1000 = metadata["score_1000"]
        self.average_score_10 = metadata["average_score_10"]
        self.average_score_100 = metadata["average_score_100"]
        self.average_score_1000 = metadata["average_score_1000"]
        self.best_score = metadata["best_score"]
        self.best_score_run = metadata["best_score_run"]

    def extract_game_area(self, frame_buffer):
        game_area_buffer = []
        for game_frame in frame_buffer.frames:
            game_area = serpent.cv.extract_region_from_image(
                game_frame.grayscale_frame,
                self.game.screen_regions["GAME_REGION"]
            )

            frame = FrameTransformer.rescale(game_area, 0.25)
            game_area_buffer.append(frame)

        return game_area_buffer
