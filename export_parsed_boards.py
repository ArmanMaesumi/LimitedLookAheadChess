import numpy as np
import chess
import chess.pgn
import time

piece_offsets = {
    chess.Piece(chess.PAWN, chess.WHITE): 0,
    chess.Piece(chess.ROOK, chess.WHITE): 64,
    chess.Piece(chess.KNIGHT, chess.WHITE): 128,
    chess.Piece(chess.BISHOP, chess.WHITE): 192,
    chess.Piece(chess.QUEEN, chess.WHITE): 256,
    chess.Piece(chess.KING, chess.WHITE): 320,

    chess.Piece(chess.PAWN, chess.BLACK): 384,
    chess.Piece(chess.ROOK, chess.BLACK): 448,
    chess.Piece(chess.KNIGHT, chess.BLACK): 512,
    chess.Piece(chess.BISHOP, chess.BLACK): 576,
    chess.Piece(chess.QUEEN, chess.BLACK): 640,
    chess.Piece(chess.KING, chess.BLACK): 704,
}


def parse_FEN(board):
    bitboard = np.zeros(775, dtype=int)
    piece_map = board.piece_map()
    for tile in piece_map:
        piece = piece_map[tile]
        index = piece_offsets[piece] + tile
        bitboard[index] = 1

    bitboard[768] = board.turn
    if board.is_check():
        if board.turn:
            bitboard[769] = 1
        else:
            bitboard[770] = 1

    bitboard[771] = board.has_kingside_castling_rights(1)
    bitboard[772] = board.has_queenside_castling_rights(1)
    bitboard[773] = board.has_kingside_castling_rights(0)
    bitboard[774] = board.has_queenside_castling_rights(0)

    return bitboard

def decompress_board(one_indices, board_len=775):
    mask = np.zeros(board_len, dtype=int)
    mask[one_indices] = 1
    return mask


def parse_pgn(pgn):
    start = time.time()
    boards = set()

    game = chess.pgn.read_game(pgn)
    num_games = 0
    num_boards = 0
    while game is not None:
        board = game.board()

        for move in game.mainline_moves():
            board.push(move)
            binary_board = parse_FEN(board)
            index_array = np.argwhere(binary_board > 0).flatten()
            boards.add(frozenset(index_array))
            num_boards += 1

        game = chess.pgn.read_game(pgn)

        if num_games == 500000:
            break

        num_games += 1

    print('Done in ', time.time() - start)
    print('Number of games: ', num_games)
    print('Number of boards: ', num_boards)
    print('Number of unique boards: ', len(boards))
    start = time.time()
    with open('boards.txt', 'a') as file:
        for board in boards:
            for i, item in enumerate(board):
                file.write(str(item))
                if i < len(board) - 1:
                    file.write(',')

            file.write('\n')
    print('Time to write: ', time.time() - start)


if __name__ == '__main__':
    parse_pgn(open("lichess_db_standard_rated_2020-02.pgn"))