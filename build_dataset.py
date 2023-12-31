#! /usr/bin/env python
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm

from gomu.gomuku import GoMuKuBoard


file_list = glob.glob("dataset/raw/gomocup2022results/Freestyle*/*.psq")
output_path = "./dataset/processed"
Path(output_path).mkdir(exist_ok=True)

print(f"Total: {len(file_list)} files.")

def get_pos(line):
    return [int(i) for i in line.split(",")[0:2]]


for index, file_path in enumerate(tqdm(file_list)):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines() 

    nrow, ncol = lines[0].split(' ')[1].strip(',').split('x')
    nrow, ncol = int(nrow), int(ncol)

    board = GoMuKuBoard(nrow=nrow, ncol=ncol, n_to_win=5)

    lines = lines[1:]

    inputs, outputs = [], []
    prev_pos = []

    to_viz = index % 1000 == 0

    winner = None
    BLACK = 1
    WHITE = -1
    DRAW = 0

    current_pos = []

    for i, line in enumerate(lines[1:]):
        if line.split(",").__len__() != 3:
            if board.is_draw():
                winner = DRAW
            elif i % 2 == 0:
                winner = WHITE
            elif i % 2 == 1:
                winner = BLACK
            break

        x, y = get_pos(line)

        if not i == 0:
            input = board.board.copy()
            output = np.zeros([nrow, ncol], dtype=np.int8)
            output[y-1, x-1] = 1

            # augmentation
            # rotate 4 x flip 3 = 12
#            for k in range(4):
#                input_rot = np.rot90(input, k=k)
#                output_rot = np.rot90(output, k=k)
#
            if len(current_pos) >= 2:
                prev_pos.append(np.stack(current_pos[-2:]))
            elif len(current_pos) == 1:
                prev_pos.append(np.stack([np.zeros([nrow, ncol]), current_pos[-1]]))
            else:
                prev_pos.append(np.zeros([2, nrow, ncol]))

            inputs.append(input)
            outputs.append(output)
            current_pos.append(output)
#
#                inputs.append(np.fliplr(input_rot))
#                outputs.append(np.fliplr(output_rot))
#
#                inputs.append(np.flipud(input_rot))
#                outputs.append(np.flipud(output_rot))
#
        if to_viz:
            print(board)

        # Update the board        
        board.set(x-1, y-1)

    # results = np.ones_like(inputs) * winner
    results = np.ones((inputs.__len__(), 1))*winner
    # print(inputs.shape)
    # print(np.ones_like(inputs).shape)
    # save dataset
    #np.savez_compressed(f"{output_path}/{index:05d}.npz", inputs=inputs, outputs=outputs, results=results)
    np.savez_compressed(f"{output_path}/{index:05d}.npz", inputs=inputs, outputs=outputs, results=results, prev_pos=prev_pos)
