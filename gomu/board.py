import numpy as np
from scipy.signal import convolve2d

# def conv2d_np(array2d, kernel):
#     kernel = np.flipud(np.fliplr(kernel)) 
#     print(kernel)
    
#     sub_matrices = np.lib.stride_tricks.as_strided(array2d,
#                                                    shape = tuple(np.subtract(array2d.shape, kernel.shape))+kernel.shape, 
#                                                    strides = array2d.strides * 2)
#     print(sub_matrices)
#     return np.einsum('ij,klij->kl', kernel, sub_matrices)

class GoMuKuBoard():
    def __init__(self, nrow, ncol, n_to_win, blank="."):
        self.board = np.zeros((nrow, ncol))
        self.nrow = nrow
        self.ncol = ncol
        self.ply = 0
        self.O = 1
        self.X = -1
        self.n_to_win = n_to_win
        self.last_player = 0
        self.last_move = None
        self.dangerous_pos = None

        self.d2m = {self.O: "O", self.X: "X", 0: blank}

    def reset(self):
        return self.__init__(nrow=self.nrow, ncol=self.ncol, n_to_win=self.n_to_win)
    
    def free_spaces(self):
        return np.argwhere(self.board.T==0).tolist()

    def whose_turn(self, ply):
        turn = self.O if ply % 2 == 0 else self.X 
        return turn
    
    def in_bounds(self, x, y):
        return self.nrow < x and self.ncol < y and x >= 0 and y >= 0

    def is_empty(self, x, y):
        return not self.board[y][x] != 0

    def total_empty(self):
        return self.board.count(0)

    # pos = (x, y) coordinate
    def set(self, x, y):
        if not self.in_bounds(x, y) and self.is_empty(x, y):
            turn = self.whose_turn(self.ply)
            if self.dangerous_pos == None or self.dangerous_pos == (x, y):
                self.board[y][x] = turn
                self.last_player = turn
                self.ply += 1
                self.last_move = (x, y)
                # reset dangerous position.
                self.dangerous_pos = None
            
            return True

        return False

    def check_with_conv2d(self, tgt_board, *kernels):
        for kernel in kernels:
            conv_calc_result = convolve2d(tgt_board, kernel, mode='valid')
            is_done = (conv_calc_result==self.n_to_win).any()
            if is_done:
                return True
        return False
            
    def check_all_cross(self, tgt_board):
        kernel1 = np.eye(self.n_to_win)
        kernel2 = np.eye(self.n_to_win)[::-1]

        return self.check_with_conv2d(tgt_board, kernel1, kernel2)
    
    def check_all_line(self, tgt_board):
        kernel1 = np.ones((1,self.n_to_win))
        kernel2 = np.ones((self.n_to_win, 1))

        return self.check_with_conv2d(tgt_board, kernel1, kernel2)

    def is_normal_play_done(self):
        turn = self.whose_turn(self.ply-1)
        me = (self.board == turn)*1
        # competitor = (self.board == -turn)*1
        return self.check_all_line(me) or self.check_all_cross(me)

    def pre_play_ending(self):
        # this scenario happened when opponent was not reacting to dangerous move.
        if self.dangerous_pos != None:
            return True, self.whose_turn(self.ply)
        
        last_x, last_y = self.last_move
        # check for 4x4
        

        # check for 3x3
        return False, None
        
    def is_draw(self):
        return self.ply >= self.nrow * self.ncol

    def is_done(self):
        return self.is_draw() or self.is_normal_play_done()

    # formatting the numpy array into human level mark.
    def forviz(self):
        data = self.board.tolist()
        formatted_data = []
        for row in data:
            formatted_row = [self.d2m[element] for element in row]
            formatted_data.append(formatted_row)
        return formatted_data

    # refactor later for the game playing with 1, 2, 3 area id
    def __repr__(self):
        formatted_data = self.forviz()
        # total_empty_space = self.total_empty()
        
        row = "    " + " | ".join([str(i) for i in range(self.ncol)]) + "\n"

        return row + "\n".join([f"{row_num} | "+" | ".join(row) for row_num, row in enumerate(formatted_data)])


if __name__ == "__main__":
    nrow = 5
    ncol = 5

    board = GoMuKuBoard(nrow=nrow, ncol=ncol, n_to_win=3)
    # bot = Nerd(-1, 5)

    for i in range(6):
        board.set(i,i)
        board.set(i,0)
    
    print(board.board)
    
    board.is_normal_play_done()
    print(board.set(0, 1))
    print(board)

    board.reset()

    scenarios = [
        [(0,0), (0, 1), (1, 1), (0, 2), (2, 2), (1, 2), (3, 3), (1, 3), (4, 4)]
    ]

    for scene in scenarios:
        for (x, y) in scene:
            board.set(x, y)
            print(board)
            print(board.is_normal_play_done())
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
