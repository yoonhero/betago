import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm

from gomu import GoMuKuBoard


file_list = glob.glob("./dataset/raw/gomocup2022results/Freestyle*/*.psq")
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

    lines = lines[1:-4]

    inputs, outputs = [], []

    to_viz = index % 1000 == 0

    for i, line in enumerate(lines[1:]):
        if line.split(",").__len__() != 3:
            break

        x, y = get_pos(line)

        if not i == 0:
            input = board.board.copy()
            output = np.zeros([ncol, nrow], dtype=np.int8)
            output[y-1, x-1] = 1

            # augmentation
            # rotate 4 x flip 3 = 12
            for k in range(4):
                input_rot = np.rot90(input, k=k)
                output_rot = np.rot90(output, k=k)

                inputs.append(input_rot)
                outputs.append(output_rot)

                inputs.append(np.fliplr(input_rot))
                outputs.append(np.fliplr(output_rot))

                inputs.append(np.flipud(input_rot))
                outputs.append(np.flipud(output_rot))

        if to_viz:
            print(board)

        # Update the board        
        board.set(x-1, y-1)

    # save dataset
    np.savez_compressed(f"{output_path}/{index:05d}.npz", inputs=inputs, outputs=outputs)
