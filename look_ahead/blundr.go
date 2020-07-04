package main

import (
	"fmt"
	"os"
	"bufio"
	"strconv"
	"github.com/ArmanMaesumi/chess"
	"flag"
	"strings"
)

func main() {
	// fmt.Println("Blundr")
	var fen string
	var depth int
	flag.StringVar(&fen, "fen", "none", "Board FEN")
	flag.IntVar(&depth, "depth", 3, "Search depth")
	flag.Parse()
	// quick_test1(6)
	if strings.Compare(fen, "none") != 0 {
		fen, _ := chess.FEN(fen)
		game := chess.NewGame(fen)
		iterative_deepening(int8(depth), game)
	} else {
		var depth int = 5
		for true {
			reader := bufio.NewReader(os.Stdin)
			fmt.Print("->")
			input, _ := reader.ReadString('\n')
			parsed_input := strings.ToLower(strings.Trim(input," \r\n"))
			if strings.Compare(parsed_input, "depth") == 0 {
				fmt.Print("depth: ")
				depth_str, _ := reader.ReadString('\n')
				depth, _ = strconv.Atoi(depth_str)
				continue
			} else {
				fen, _ := chess.FEN(input)
				game := chess.NewGame(fen)
				iterative_deepening(int8(depth), game)
			}
			
		}
	}
}

func quick_test1(depth int8) {
	fen, _ := chess.FEN("2kr4/pp1n1p2/q1p1p3/2P5/1P1PPbp1/P1N3Pr/6K1/R2Q1R2 w - - 0 27")
	game := chess.NewGame(fen)
	iterative_deepening(depth, game)
}

func iterative_deepening(max_depth int8, game *chess.Game) {
	maximize := true
	if game.Position().Turn() == chess.Black {
		maximize = false
	}

	for i := int8(2); i < max_depth; i++ {
		minimax_root(i, game, maximize, false)
	}
}
