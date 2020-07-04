package main

import(
	"fmt"
	"time"
	"github.com/ArmanMaesumi/chess"
	"math"
	"sort"
	"sync/atomic"
	"golang.org/x/sync/syncmap"
)

func garbage(){
	fmt.Printf("")
}

type option struct {
	move *chess.Move
	eval float64
}

type t_entry struct {
	depth 	int8
	flag	int8
	eval	float64
	best	*chess.Move
}

// var transposition = make(map[[16]byte]t_entry)
var transposition = syncmap.Map{}
var sync_eval_cache = syncmap.Map{}
var eval_cache = make(map[[16]byte]float64)
var cache_hits uint64
var table_hits uint64
var nodes uint64

var quick_prune bool = false

func minimax_root(depth int8, game *chess.Game, maximizing bool, verbose bool) string{
	cache_hits = 0
	table_hits = 0
	nodes = 0
	// move_order(game.Position(), maximizing)

	start := time.Now()

	pos := game.Position()
	// legal_moves := game.ValidMoves()
	sorted_moves := move_order(pos, maximizing)

	var final_move string
	// var value float64

	chans := make([]chan float64, len(sorted_moves))
	options := make([]option, len(sorted_moves))

	for i := range chans {
		chans[i] = make(chan float64)
	}

	if maximizing {
		for i, option := range sorted_moves {
			new_pos := pos.Update(option.move)
			atomic.AddUint64(&nodes, 1)
			options[i].move = option.move
			go conc_minimax(depth - 1, new_pos, -9999, 9999, !maximizing, chans[i], true)
		}

		for i := range chans {
			options[i].eval = <-chans[i]
			// fmt.Println(sorted_moves[i].move, ": ", <-chans[i])
		}
		
		sort.Slice(options, func(p, q int) bool {  
			return options[p].eval > options[q].eval })

		// for i, option := range sorted_moves {
		// 	if quick_prune && i >= len(sorted_moves)/2 {
		// 		break
		// 	}

		// 	new_pos := pos.Update(option.move)
		// 	value = math.Max(best_move, minimax(depth - 1, new_pos, -9999, 9999, !maximizing))
		// 	if value > best_move{
		// 		best_move = value
		// 		final_move = option.move.String()
		// 	}
		// }
	} else {
		for i, option := range sorted_moves {
			new_pos := pos.Update(option.move)
			atomic.AddUint64(&nodes, 1)
			options[i].move = option.move
			go conc_minimax(depth - 1, new_pos, -9999, 9999, !maximizing, chans[i], true)
		}

		for i := range chans {
			options[i].eval = <-chans[i]
		}
		
		sort.Slice(options, func(p, q int) bool {  
			return options[p].eval < options[q].eval })

		// for _, option := range sorted_moves {
		// 	new_pos := pos.Update(option.move)
		// 	nodes++
		// 	value = math.Min(best_move, minimax(depth - 1, new_pos, -9999, 9999, !maximizing))
		// 	if value < best_move{
		// 		best_move = value
		// 		final_move = option.move.String()
		// 	}
		// }
	}

	if verbose {
		elapsed := time.Now().Sub(start).Seconds()
		fmt.Println("Elapsed time: ", elapsed)
		fmt.Println("Cache hits: ", cache_hits)
		fmt.Println("Table hits: ", table_hits)
		fmt.Println("Nodes: ", nodes)
		fmt.Println("nps: ", float64(nodes)/elapsed)
		for i := range options {
			if i >= 3 {
				break
			}
			fmt.Println(options[i].move, ": ", options[i].eval)
		}
	} else {
		fmt.Println()
		fmt.Print(options[0].move)
	}

	return final_move
}

func minimax(depth int8, pos *chess.Position, alpha float64, beta float64, maximizing bool) float64{
	status := pos.Status()
	switch status {
	case chess.Checkmate:
		if pos.Turn() == chess.White {
			return -99999 - float64(depth)
		} else {
			return 99999 + float64(depth)
		}
	case chess.Stalemate:
		return 0
	case chess.FivefoldRepetition:
		return 0
	}

	if depth == 0 {
		return evaluate(pos)
	}

	sorted_moves := move_order(pos, maximizing)
	var best_move float64

	if maximizing {
		best_move = -9999

		for i, option := range sorted_moves {
			if quick_prune && i >= len(sorted_moves)/2 {
				break
			}
			new_pos := pos.Update(option.move)
			best_move = math.Max(best_move, minimax(depth - 1, new_pos, alpha, beta, !maximizing))
			alpha = math.Max(alpha, best_move)
			if beta <= alpha{
				break
			}
		}
	} else {
		best_move = 9999

		for i, option := range sorted_moves {
			if quick_prune && i >= len(sorted_moves)/2 {
				break
			}
			new_pos := pos.Update(option.move)
			best_move = math.Min(best_move, minimax(depth - 1, new_pos, alpha, beta, !maximizing))
			beta = math.Min(beta, best_move)
			if beta <= alpha{
				break
			}
		}
	}

	return best_move
}

func q_search(depth int8, pos *chess.Position, alpha float64, beta float64, maximizing bool) float64 {
	stand_pat := evaluate_single(pos)
	if depth == 0 {
		return stand_pat
	}
	if stand_pat >= beta {
		return beta;
	}

	if alpha < stand_pat {
		alpha = stand_pat
	}

	legal_moves := pos.ValidMoves()
	var best_move float64

	if maximizing {
		best_move = -9999

		for _, move := range legal_moves {
			if move.HasTag(chess.Capture) {
				new_pos := pos.Update(move)
				atomic.AddUint64(&nodes, 1)
				best_move = math.Max(best_move, q_search(depth - 1, new_pos, alpha, beta, !maximizing))
				alpha = math.Max(alpha, best_move)
				if beta <= alpha{
					break
				}
			}
		}
	} else {
		best_move = 9999

		for _, move := range legal_moves {
			if move.HasTag(chess.Capture) {
				new_pos := pos.Update(move)
				atomic.AddUint64(&nodes, 1)
				best_move = math.Min(best_move, q_search(depth - 1, new_pos, alpha, beta, !maximizing))
				beta = math.Min(beta, best_move)
				if beta <= alpha{
					break
				}
			}
		}
	}

	return best_move
}

func conc_minimax(depth int8, pos *chess.Position, alpha float64, beta float64, 
					maximizing bool, c chan float64, update_chan bool) float64{

	status := pos.Status()
	switch status {
	case chess.Checkmate:
		if pos.Turn() == chess.White {
			return -99999 - float64(depth)
		} else {
			return 99999 + float64(depth)
		}
	case chess.Stalemate:
		return 0
	case chess.FivefoldRepetition:
		return 0
	}

	alpha_orig := alpha
	hash := pos.Hash()

	entry, ok := transposition.Load(hash)
	
	if ok {
		trans_entry := entry.(t_entry)
		if trans_entry.depth >= depth {
			table_hits++
			if trans_entry.flag == 0 {
				return trans_entry.eval
			} else if trans_entry.flag == 1 {
				beta = math.Min(beta, trans_entry.eval)
			} else if trans_entry.flag == 2 {
				alpha = math.Max(alpha, trans_entry.eval)
			}
	
			if alpha >= beta {
				return trans_entry.eval
			}
		}
	}

	if depth == 0 {
		return evaluate(pos) 
		//return q_search(3, pos, alpha, beta, maximizing)
	}

	sorted_moves := move_order(pos, maximizing)
	var best_move float64

	if maximizing {
		best_move = -9999

		for _, option := range sorted_moves {
			new_pos := pos.Update(option.move)
			atomic.AddUint64(&nodes, 1)
			best_move = math.Max(best_move, conc_minimax(depth - 1, new_pos, alpha, beta, !maximizing, c, false))
			alpha = math.Max(alpha, best_move)
			if beta <= alpha{
				break
			}
		}
	} else {
		best_move = 9999

		for _, option := range sorted_moves {
			new_pos := pos.Update(option.move)
			atomic.AddUint64(&nodes, 1)
			best_move = math.Min(best_move, conc_minimax(depth - 1, new_pos, alpha, beta, !maximizing, c, false))
			beta = math.Min(beta, best_move)
			if beta <= alpha{
				break
			}
		}
	}

	new_entry := t_entry{
		depth: depth,
		eval: best_move,
	}

	if best_move <= alpha_orig {
		new_entry.flag = 1
	} else if best_move >= beta {
		new_entry.flag = 2
	} else {
		new_entry.flag = 0
	}

	transposition.Store(hash, new_entry)

	if update_chan {
		c <- best_move
	}

	return best_move
}

func evaluate(pos *chess.Position) float64 {
	hash := pos.Hash()

	val, ok := sync_eval_cache.Load(hash)
	if ok {
		cache_hits++
		return val.(float64)
	}
	// if val, ok := eval_cache[hash]; ok {
	// 	cache_hits++
	// 	return val
	// }
	// var eval float64 = 0
	// fmt.Println(game.FEN())
	bitboard := parse_pos(pos)
	eval := float64(predict(bitboard)[0][0])
	sync_eval_cache.Store(pos.Hash(), eval)

	move_order(pos, false)

	val, ok = sync_eval_cache.Load(hash)
	return val.(float64)
	// return eval_cache[hash]
}

func evaluate_single(pos *chess.Position) float64 {
	hash := pos.Hash()

	val, ok := sync_eval_cache.Load(hash)
	if ok {
		cache_hits++
		return val.(float64)
	}
	
	bitboard := parse_pos(pos)
	eval := float64(predict(bitboard)[0][0])
	sync_eval_cache.Store(pos.Hash(), eval)

	return eval
}

func move_order(pos *chess.Position, maximizing bool) []option{ 
	legal_moves := pos.ValidMoves()
	options := make([]option, len(legal_moves))

	var pboards [][775]float32
	var pboard_option_index []int
	var pboard_hashes [][16]byte

	for i, move := range legal_moves {
		new_pos := pos.Update(move)
		hash := new_pos.Hash()

		val, ok := sync_eval_cache.Load(hash)
		if ok {
			cache_hits++

			options[i].move = move
			options[i].eval = val.(float64)
		} else {
			pboards = append(pboards, parse_pos(new_pos))
			pboard_option_index = append(pboard_option_index, i)
			pboard_hashes = append(pboard_hashes, hash)
			
			options[i].move = move
		}

		// if val, ok := eval_cache[hash]; ok {
		// 	cache_hits++

		// 	options[i].move = move
		// 	options[i].eval = val
		// } else {
		// 	pboards = append(pboards, parse_pos(new_pos))
		// 	pboard_option_index = append(pboard_option_index, i)
		// 	pboard_hashes = append(pboard_hashes, hash)
			
		// 	options[i].move = move
		// 	options[i].eval = 0.0
		// }
	}

	if len(pboards) > 0 {
		pboard_evals := predict_batch(pboards)
		for i, idx := range pboard_option_index {
			eval := float64(pboard_evals[i][0])
			options[idx].eval = eval
			sync_eval_cache.Store(pboard_hashes[i], eval)
			// eval_cache[pboard_hashes[i]] = eval
		}
	}

	if maximizing {
		sort.Slice(options, func(p, q int) bool {  
			return options[p].eval > options[q].eval }) 
	} else {	
		sort.Slice(options, func(p, q int) bool {  
			return options[p].eval < options[q].eval }) 
	}

	return options

	// for _, option := range options {
	// 	fmt.Println(option.move, ", ", option.eval)
	// }

}