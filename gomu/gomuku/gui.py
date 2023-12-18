import itertools
import pygame
import random
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import seaborn as sns
import pylab

from .board import GoMuKuBoard
from .utils import open_record, RECORD_KEY, GAME_INFO_KEY
from .errors import BotError, PosError

from ..helpers import DEBUG

class Colors:
    BLACK = 0, 0, 0
    WHITE = 255, 255, 255
    BROWN = 205, 128, 0


class GomokuGUI:
    def __init__(
        self,
        size=60,
        rows=15,
        cols=15,
        n_to_win=5,
        bot = None,
        viz_data = None
    ):
        self.rows = rows
        self.cols = cols
        self.w = rows * size
        self.h = cols * size
        self.plot_h = self.h/4
        self.size = size
        self.piece_size = size / 2
        self.half_size = size // 2
        pygame.init()
        pygame.display.set_caption("AlphaGomu")
        self.screen = pygame.display.set_mode((self.w, self.h+self.plot_h))
        self.screen.fill(Colors.WHITE)
        self.player_colors = {1: Colors.WHITE, 0: Colors.BLACK}
        self.player_names = {1: "White", 0: "Black"}
        self.board = GoMuKuBoard(rows, cols, n_to_win)

        self.values_for_plotting = []
        
        self.with_human = bot == None
        self.bot = bot

        if viz_data != None:
            self.visualize(viz_data)

    def row_lines(self):
        half = self.half_size

        for y in range(half, self.h - half + self.size, self.size):
            yield (half, y), (self.w - half, y)

    def col_lines(self):
        half = self.half_size

        for x in range(half, self.w - half + self.size, self.size):
            yield (x, half), (x, self.h - half)
        
    def draw_background(self):
        rect = pygame.Rect(0, 0, self.w, self.h)
        pygame.draw.rect(self.screen, Colors.BROWN, rect)

    def draw_lines(self):
        lines = itertools.chain(self.col_lines(), self.row_lines())

        for start, end in lines:
            pygame.draw.line(
                self.screen, 
                Colors.BLACK, 
                start, 
                end, 
                width=2
            )

    def draw_board(self):
        self.draw_background()
        self.draw_lines()

    def draw_control_panel(self):   
        font = pygame.font.Font("./gomu/MaruBuri-Regular.ttf", self.w//10)
        next_label = font.render("NEXT", True, Colors.WHITE, Colors.BLACK)
        prev_label = font.render("PREV", True, Colors.WHITE, Colors.BLACK)
        cent_y = self.h // 2 - next_label.get_height() // 2
        next_label_pos = (self.w-10-next_label.get_width(), cent_y)
        prev_label_pos = (10+prev_label.get_width(), cent_y)
        self.screen.blit(next_label, (self.w-10-next_label.get_width(), cent_y))
        self.screen.blit(prev_label, (10+prev_label.get_width(), cent_y))
        return next_label_pos, prev_label_pos
        
    def draw_piece(self, row, col):
        player = self.board.last_player
        circle_pos = (
           col * self.size + self.half_size, 
           row * self.size + self.half_size,
        )
        pygame.draw.circle(
           self.screen, 
           self.player_colors[player], 
           circle_pos, 
           self.piece_size
        )

    def show_outcome(self):
        player = self.player_names[self.board.last_player]
        msg = "draw!" if self.board.is_draw() else f"{player} wins!"
        font_size = self.w // 10
        font = pygame.font.Font("./gomu/MaruBuri-Regular.ttf", font_size)
        label = font.render(msg, True, Colors.WHITE, Colors.BLACK)
        x = self.w // 2 - label.get_width() // 2
        y = self.h // 2 - label.get_height() // 2
        self.screen.blit(label, (x, y))

    def exit_on_click(self):
        while True:
            for event in pygame.event.get():
                if (event.type == pygame.QUIT or 
                        event.type == pygame.MOUSEBUTTONDOWN):
                    pygame.quit()
                    return
    
    def is_human_turn(self):
        return self.board.ply % 2 != int(self.is_human_first)

    def update_board(self, col, row):
        if self.board.set(col, row):
            self.draw_piece(col=col, row=row)
            self.update_plot()
            pygame.display.update()
            return True
        return False

    def ai_turn(self):
        # 1 => first, -1 => later
        if self.is_human_turn():
            return False
        
        board_state = self.board.board
        next_pos, winning_percentage = self.bot(board_state, turn=int(self.is_human_first))
        col, row = next_pos[0]  
        self.values_for_plotting.append(1-winning_percentage.item())

        if DEBUG >= 2:
            print(f"Your WINNING Percentage: {1-winning_percentage.item()}")

        if not self.update_board(col, row):
            raise Exception([BotError, PosError(col, row)])

    def make_move(self, x, y):
        col = x // self.size
        row = y // self.size

        # validate the human illegal movement
        if not self.is_human_turn():
            self.ai_turn()
            return
        
        if self.update_board(col, row):
            if not self.with_human and not self.board.is_gameover():
                self.ai_turn()  

    def initiate_game(self):
        pygame.time.Clock().tick(60)  
        self.draw_board()
        pygame.display.update()

    def play(self):
        self.initiate_game()

        if not self.with_human:
            # Random Starting
            self.is_human_first = random.random() > 0.5
            self.bot.set_turn(int(self.is_human_first))
            if not self.is_human_first:
                self.ai_turn()

        while not self.board.is_gameover() and not self.board.is_draw():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.make_move(*event.pos)
                    pygame.display.update()
                    
        self.show_outcome()
        pygame.display.update()
        self.exit_on_click()

    def update_plot(self):
        screen = pygame.display.get_surface()
        surf = self.drawing_plot()
        screen.blit(surf, (0, self.h))

    def drawing_plot(self):
        w, h = self.w/96, self.plot_h/96
        fig = pylab.figure(figsize=[w,h],
                   dpi=100,
                   )
        ax = fig.gca()
        ax.plot(self.values_for_plotting)
        ax.set_ylim(0, 1)

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()

        size = canvas.get_width_height()
        w, _ = size

        surf = pygame.image.fromstring(raw_data, size, "RGB")

        # align_pos = (self.half_size-w//2, self.h)
        return surf

    @classmethod
    def lets_viz(cls, record_paths):
        # Please give me the same game info scenarios to avoid the error
        records = [open_record(record_path) for record_path in record_paths]
        game_info = records[-1][GAME_INFO_KEY]
        viz_data = [record[RECORD_KEY] for record in records]
        nrow, ncol, n_to_win = game_info["nrow"], game_info["ncol"], game_info["n_to_win"]

        return cls(rows=nrow, cols=ncol, n_to_win=n_to_win, viz_data=viz_data)

    # def visualize(self, viz_data):
    #     self.initiate_game()

    #     prev_pos, next_pos = self.draw_control_panel()
    #     ok_range = 10

    #     for viz in viz_data:
    #         cur = 0
    #         self.update_board()
    #         while True:
    #             for event in pygame.event.get():
    #                 if event.type == pygame.MOUSEBUTTONDOWN:
    #                     x, y = event.pos
    #                     if (x>prev_pos[0]-ok_range or x<prev_pos[0]+ok_range) and (y>prev_pos[1]-ok_range and y<prev_pos[1]+ok_range):


if __name__ == "__main__":
    game = GomokuGUI(rows=5, cols=5, n_to_win=4)
    game.play()