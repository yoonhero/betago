import numpy as np
from scipy.signal import convolve2d

from ..helpers import DEBUG

class GoMuKuBoard():
    def __init__(self, nrow, ncol, n_to_win, blank="."):
        # Black -> White
        self._board = np.zeros((2, nrow, ncol))
        self.nrow = nrow
        self.ncol = ncol
        # The number of stone in the board
        self.ply = 0
        self.STONE = 1
        self.n_to_win = n_to_win
        self.last_player = 0
        self.last_move = None

        self.BLACK = "O"
        self.WHITE = "X"
        self.BLANK = " "
        self.s2m = {-1: self.BLACK, 1: self.WHITE, 0: self.BLANK}

    @property
    def board(self):
        return self._board.copy()

    def reset(self):
        return self.__init__(nrow=self.nrow, ncol=self.ncol, n_to_win=self.n_to_win)
    
    # Getting Free space list of x, y coordination
    def free_space_coordination(self) -> list[list[int]]:
        not_free_space = self._board.sum(0)
        free_space_coordination = np.argwhere(not_free_space.T==0).tolist()
        return free_space_coordination
    
    def not_free_space(self):
        not_free_space = self._board.sum(0)
        return not_free_space

    def whose_turn(self, ply):
        # turn = self.O if ply % 2 == 0 else self.X 
        return ply % 2
    
    def in_bounds(self, x, y):
        return self.nrow < x and self.ncol < y and x >= 0 and y >= 0

    def is_empty(self, x, y):
        # return not (self._board.sum(0)[y][x] != 0).any()
        return self._board.sum(0)[y][x] == 0

    def total_empty(self):
        return self._board.count(0)

    # pos = (x, y) coordinate
    def set(self, x, y):
        if not self.in_bounds(x, y) and self.is_empty(x, y):
            turn = self.whose_turn(self.ply)
            self._board[turn][y][x] = self.STONE
            self.last_player = turn
            self.ply += 1
            self.last_move = (x, y)
            
            if DEBUG >= 2:
                print(f"Player {turn} plays {self.last_move}!")
            if DEBUG >= 3:
                print(self._board)
        
            return True
            
        return False

    def check_with_conv2d(self, tgt_board, *kernels):
        for kernel in kernels:
            conv_calc_result = convolve2d(tgt_board, kernel, mode='valid')
            is_done = (conv_calc_result==self.n_to_win).any()
            if is_done:
                return True
        return False
            
    def check_all_diagonal(self, tgt_board):
        kernel1 = np.eye(self.n_to_win)
        kernel2 = np.eye(self.n_to_win)[::-1]

        return self.check_with_conv2d(tgt_board, kernel1, kernel2)
    
    def check_all_wh(self, tgt_board):
        kernel1 = np.ones((1,self.n_to_win))
        kernel2 = np.ones((self.n_to_win, 1))

        return self.check_with_conv2d(tgt_board, kernel1, kernel2)

    def is_gameover(self):
        turn = self.whose_turn(self.ply-1)
        tgt_board = self._board[turn]
        return self.check_all_wh(tgt_board) or self.check_all_diagonal(tgt_board)
        
    def is_draw(self):
        return self.ply >= self.nrow * self.ncol

    # formatting the numpy array into human level.
    def forviz(self):
        data = self._board
        formatted_data = []

        tmp = np.zeros((self.nrow, self.ncol))
        for stone_color in range(2):
            tmp[data[stone_color] != 0] = 1-2*stone_color
        formatted_data = tmp.tolist()
        formatted_data = [[self.s2m[int(item)] for item in row] for row in formatted_data]
    
        return formatted_data

    def __repr__(self):
        formatted_data = self.forviz()
       
        row = "    " + " | ".join([str(i) for i in range(self.ncol)]) + "\n"

        return row + "\n".join([f"{row_num} | "+" | ".join(row) for row_num, row in enumerate(formatted_data)])



if __name__ == "__main__":
    nrow = 5
    ncol = 5

    board = GoMuKuBoard(nrow=nrow, ncol=ncol, n_to_win=3)
    # bot = Nerd(-1, 5)

    for i in range(4):
        board.set(i,i)
        ok = board.set(i,0)
        print(ok)
    
    print(board.board)
    
    board.is_gameover()
    print(board.set(0, 1))
    print(board)

    board.reset()

    scenarios = [
        [(0,0), (0, 1), (1, 1), (0, 2), (2, 2), (1, 2), (3, 3), (1, 3), (4, 4)]
    ]

    # for scene in scenarios:
    #     for (x, y) in scene:
    #         board.set(x, y)
    #         print(board)
    #         print(board.is_normal_play_done())
    # print(np.where(board.board==0)*nrow+np.stack([list(range(ncol))*nrow]))
    
    # while True:
    #     inp = input("> YOU: x, y ")
    #     x, y = [int(pos) for pos in inp.split(",")]
    #     # if player set a stone in not available space.
    #     if not board.set(x, y):
    #         continue

    #     freeee = np.argwhere(board.board==0).tolist()
    #     # selected_pos = bot(board.board, freeee)
    #     selected_x, selected_y = selected_pos
    #     board.set(selected_x, selected_y)
    #     print(board)
