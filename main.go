package main

import (
	"firtsNeural/mpl"
	"firtsNeural/operations"
	"flag"
	"fmt"
)

func main() {
	net := mpl.CreateNetwork(784, 500, 10, 01)

	mnist := flag.String("mnist", "", "Either train or predict to evaluate neural network")
	file := flag.String("file", "", "File name of 28 x 28 PNG file to evaluate")
	flag.Parse()
	switch *mnist {
	case "train":
		net.Load()
		net.TrainNetwork()
		net.Save()
	case "predict":
		net.Load()
		net.Check()
	}
	if *file != "" {
		net.Load()
		image := operations.GetImage(*file)
		operations.PrintImage(*file)
		fmt.Println("prediction:", net.PredictFromImage(image))
	}
}
