import numpy as np

from bot import *

turns = {"w": 1, "b": -1}

# Hyper-Parameters
ngyms = 10
nrow = 3
ncol = 3
n_to_win = 3

competitor_turn = "w"
learner_turn = "b"
learner = Nerd(turns[learner_turn], n_to_win=n_to_win)

kwargs = {"RandomMover": {}, "Nerd": {"turn": turns[competitor_turn], "n_to_win": n_to_win}}
rl_env = RLEnv(ngyms=ngyms, nrow=nrow, ncol=ncol, n_to_win=n_to_win, competitor="Nerd", viz_training_status=False, viz_gym=False, engine_params=kwargs["Nerd"])
n_epochs = 10

gym_id = 0
for episode in range(n_epochs):
    while True:
        state, free_spaces = rl_env.current_state(gym_id)

        x, y = learner(state, free_spaces)

        episode_summary = rl_env.proceed(x, y, gym_id=gym_id)

        print(episode_summary)

        if episode_summary["done"]:
            break

    