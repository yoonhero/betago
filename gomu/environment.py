import numpy as np
import json
from pathlib import Path
import time
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from copy import deepcopy

from .board import GoMuKuBoard
from .gui import GomokuGUI
from .utils import save_record
from .errors import BoardError, BotError, RLEnvError
from .errors import ThreadingErrorHandler


class RLEnv():
    # competitor: None, RandomMover, Nerd
    def __init__(self, ngyms, nrow, ncol, n_to_win, viz_gym=False, viz_training_status=False, engine=None):
        # environment's requirements
        # start episode
        # manage the flow of the game -> give a environment state for rl agent learning.
        # give a reward for an appropriate action after the game or during the game
        # end episode
        # renewal the game
        self.ngyms = ngyms
        self.nrow = nrow
        self.ncol = ncol
        self.n_to_win = n_to_win
        self.gyms: list[GoMuKuBoard] = [GoMuKuBoard(nrow=nrow, ncol=ncol, n_to_win=n_to_win) for _ in range(ngyms)]
        self.game_info = {"ncol": self.ncol, "nrow": self.nrow, "n_to_win": self.n_to_win}
        self.n_episode = self.initialize_empty_list(ngyms, 0)
        self.episodes = self.initialize_empty_list(ngyms, list())

        self.records = self.initialize_empty_list(ngyms, list())
        Path("./tmp").mkdir(exist_ok=True)

        # Only save the winning position(?)
        self.results = self.initialize_empty_list(ngyms, list())

        self.is_competitor_exist = engine != None
        #engines = {"RandomMover": RandomMover, "Nerd": Nerd}
        
        if self.is_competitor_exist != None:
            #self.engine = engines[competitor](**engine_params)
            self.engine = engine

            # TODO: Extend API design for parallel learning pipeline
            if self.gyms[0].whose_turn(self.gyms[0].ply) == self.engine.turn:
                self.competitor_turn(0)


        self.bucket = queue.Queue()
        if viz_gym:
            # Training Visualization with Treading.
            self.viz_game_thread = ThreadingErrorHandler(bucket=self.bucket, target=self.viz_gym)
            self.viz_game_thread.start()
        
        if viz_training_status:
            self.viz_training_status_thread = ThreadingErrorHandler(bucket=self.bucket, target=self.viz_training_status)
            self.viz_training_status_thread.start()

    def init_game(self, gym_id):
        self.gyms[gym_id].reset()
        self.n_episode[gym_id] += 1
        
    def get_reward(self, gym_id):
        # win 10
        # lose -10
        # draw 0
        if self.gyms[gym_id].is_draw():
            return 0

        # add scenarios to penalty ridiculous movement.
        return 10

    def initialize_empty_list(self, length, item):
        return [deepcopy(item) for _ in range(length)]

    # Return True for sucess proceeding, vice versa.
    def next(self, x, y, gym_id):
        # set the stone with checking.
        if self.gyms[gym_id].set(x, y):
            self.update_history((x, y), gym_id)
            return True
        return False

    def competitor_turn(self, gym_id):
        state, free_spaces = self.current_state(gym_id)
        x, y = self.engine(state, free_spaces)
        return self.next(x, y, gym_id=gym_id)
    
    def current_state(self, gym_id):
        state = self.gyms[gym_id].board
        free_spaces = self.gyms[gym_id].free_spaces()
        return state, free_spaces

    def proceed(self, x, y, gym_id):
        if not self.next(x, y, gym_id=gym_id):
            raise BoardError()
        
        if self.is_competitor_exist and not self.gyms[gym_id].is_done():
            if not self.competitor_turn(gym_id):
                raise BoardError()

        # check for the ending
        episode_summary = {"reward":0, "state": None, "n_episode": self.n_episode[gym_id], "ply": self.gyms[gym_id].ply, "done": None}
        episode_summary['state'] = self.gyms[gym_id].board
        is_done = self.gyms[gym_id].is_done()
        episode_summary["done"] = is_done

        if is_done:
            last_player = self.gyms[gym_id].last_player
            self.results[gym_id].append(last_player)

            reward = self.get_reward(gym_id)
            episode_summary["reward"] = reward
            
            # save the record
            self._save_record(gym_id)

            # reset the game
            self.init_game(gym_id)

        return episode_summary
    
    def update_history(self, pos, gym_id):
        cur_episode = self.n_episode[gym_id]

        if len(self.records[gym_id]) <= cur_episode:
            self.records[gym_id].append(list())
        self.records[gym_id][cur_episode].append(pos)
    
    def plot_result(self, gym_id, who, figure, axes):
        tgt_result = np.array(self.gyms[gym_id])
        tgt_result[tgt_result!=who] = 0
        tgt_result = np.abs(tgt_result)

        total_games = len(tgt_result)
        winning = np.sum(tgt_result)
        winning_percentage = f"{(winning / total_games):.2f}"
        x = np.arange(2)
        y = [winning, total_games-winning]
        cumulative_sum_of_winning = np.cumsum(tgt_result)

        figure.suptitle(f"Training Status of {self.n_episode[gym_id]} epoch in gym id {gym_id}")
        axes[0].bar(x, y)
        axes[0].xticks(x, ["Win", "Lose"])
        axes[1].plot(cumulative_sum_of_winning)
        axes[1].xlabel("Epoch")
        axes[1].ylabel("Winning Games")
    
    def _save_record(self, gym_id):
        formatted_pos = lambda x: f"{x[0]},{x[1]}"

        cur_episode = self.n_episode[gym_id]
        tmp = self.records[gym_id][cur_episode]
        formatted_record = [formatted_pos(pos) for pos in tmp]

        return save_record(cur_episode, self.game_info, formatted_record)

    def is_visualize_no_problem(self):
        try: 
            exc = self.bucket.get(block=False)
        except queue.Empty:
            pass
        else:
            exc_type, exc_obj, exc_trace = exc
            print(exc_type, exc_obj, exc_trace)
            return False
        
        self.viz_game_thread.join(0.1)
        return self.viz_game_thread.isAlive()
    
    def get_random_gym_id(self):
        return random.randint(0, self.ngyms)

    # Show Viz Using GUI Class
    def viz_gym(self):
        print("?///")
        gui_tool = GomokuGUI(rows=self.nrow, cols=self.ncol, n_to_win=self.n_to_win, viz=True)

        gui_tool.initiate_game()

        random_gym_id = self.get_random_gym_id()
        cur_board = self.gyms[random_gym_id].board
        while True:
            now_board = self.gyms[random_gym_id].board
            if cur_board != now_board:
                diff = cur_board - now_board
                indices = np.argwhere(diff!=0).tolist()
                if len(indices) != 1:
                    gui_tool.initiate_game()
                    cur_board = np.zeros((self.nrow, self.ncol))
                    continue
                
                col, row = indices[0]
                gui_tool.update_board(col, row)
                time.sleep(0.1)

    def viz_training_status(self, who):
        random_gym_id = self.get_random_gym_id()
        fig, ax = plt.subplots(1, 2)
        animation.FuncAnimation(fig, self.plot_result, interval=1000, gym_id=random_gym_id, who=who, figure=fig, axes=ax)
    
    def __repr__(self) -> str:
        return self.game_env
