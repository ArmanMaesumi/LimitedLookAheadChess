package main

import (
	"github.com/ArmanMaesumi/chess"
	tg "github.com/galeone/tfgo"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var piece_offsets = map[int8]int16{
	6: 0,
	3: 64,
	5: 128,
	4: 192,
	2: 256,
	1: 320,
	12: 384,
	9: 448,
	11: 512,
	10: 576,
	8: 640,
	7: 704,
}

var model = tg.LoadModel("static_evaluation_model", []string{"myTag"}, nil)

func predict(bitboard [775]float32) [][]float32{
	var batch [1][775]float32
	batch[0] = bitboard

	input, _ := tf.NewTensor(batch)
	results := model.Exec([]tf.Output{
			model.Op("activation_4/Tanh", 0),
	}, map[tf.Output]*tf.Tensor{
			model.Op("encoder_0", 0): input,
	})

	predictions := results[0].Value().([][]float32)

	return predictions
}

func predict_batch(bitboards [][775]float32) [][]float32 {
	input, _ := tf.NewTensor(bitboards)
	results := model.Exec([]tf.Output{
			model.Op("activation_4/Tanh", 0),
	}, map[tf.Output]*tf.Tensor{
			model.Op("encoder_0", 0): input,
	})

	predictions := results[0].Value().([][]float32)

	return predictions
}

func parse_pos(pos *chess.Position) [775]float32 {
	square_map := pos.Board().SquareMap()
	var bitboard [775]float32

	for square, piece := range square_map {
		bitboard[piece_offsets[int8(piece)] + int16(square)] = 1
	}

	bitboard[768] = float32(pos.Turn()) - 1

    if pos.InCheck() {
        if pos.Turn() == chess.White {
            bitboard[769] = 1
		} else {
			bitboard[770] = 1
		}
	}
	
	bitboard[771] = BoolToFloat32(pos.CastleRights().CanCastle(chess.White, chess.KingSide))
	bitboard[772] = BoolToFloat32(pos.CastleRights().CanCastle(chess.White, chess.QueenSide))
	bitboard[773] = BoolToFloat32(pos.CastleRights().CanCastle(chess.Black, chess.KingSide))
	bitboard[774] = BoolToFloat32(pos.CastleRights().CanCastle(chess.Black, chess.QueenSide))

	return bitboard
}

func BoolToFloat32(b bool) float32 {
	if b {
		return 1.0
	}

	return 0.0
}