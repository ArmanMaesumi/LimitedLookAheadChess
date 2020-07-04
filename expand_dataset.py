import chess
import chess.engine
import numpy as np
import random
import os


class ExpandDataset:
    def __init__(self,
                 stockfish_exe,
                 new_dict_filename,
                 threads,
                 score_threshold,
                 expansion_factor,
                 export_period):

        if not str(new_dict_filename).endswith('.npy'):
            new_dict_filename += '.npy'

        self.stockfish_exe = stockfish_exe
        self.new_dict_filename = new_dict_filename
        self.threads = int(threads)
        self.score_threshold = score_threshold
        self.expansion_factor = expansion_factor
        self.export_period = export_period
        self.score_dict = {}

        self.import_hash_table('scores(20million).npy')

    def import_hash_table(self, dict_file):
        # Attempt to load already processed boards
        if os.path.isfile(dict_file):
            self.score_dict = np.load(dict_file).item()
            print('Imported hash table of length {}', str(len(self.score_dict)))
        else:
            print('No hash table found. Creating new hash table.')

    def export_hash_table(self, table):
        np.save(self.new_dict_filename, table)

    def evaluate_random_moves(self):
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_exe)
        engine.configure({"Threads": 1, "Hash": 64})

        new_position_dict = {}
        new_positions = 0
        for i, position in enumerate(self.score_dict):
            # print(position)
            score = self.score_dict[position]
            if abs(score) <= self.score_threshold:
                board = chess.Board(position)
                moves = [b for b in board.generate_legal_moves()]

                if len(moves) < self.expansion_factor:
                    moves = random.sample(moves, k=len(moves))
                else:
                    moves = random.sample(moves, k=self.expansion_factor)

                # original_eval = engine.analyse(board, chess.engine.Limit(time=0.300)).score.white().score(mate_score=50000)
                for move in moves:
                    board.push(move)
                    if board.fen() in self.score_dict or board.fen() in new_position_dict:
                        board.pop()
                        continue

                    evaluation = engine.analyse(board, chess.engine.Limit(time=0.300))
                    new_position_dict[board.fen()] = evaluation.score.white().score(mate_score=50000)
                    new_positions += 1
                    board.pop()
                    # print('From: {} To: {}'.format(score, evaluation.score.white().score(mate_score=50000)))

                    if new_positions % self.export_period == 0 and new_positions > 0:
                        print('Exporting with length: ' + str(len(new_position_dict)))
                        self.export_hash_table(new_position_dict)

        print('Exporting with length: ' + str(len(new_position_dict)))
        self.export_hash_table(new_position_dict)

if __name__ == '__main__':
    filename = input('Export filename: ')
    dataset_expander = ExpandDataset(stockfish_exe='Stockfish\\stockfish_20011801_x64_bmi2.exe',
                                     new_dict_filename=filename,
                                     threads=1,
                                     score_threshold=150,
                                     expansion_factor=5,
                                     export_period=25000)

    dataset_expander.evaluate_random_moves()