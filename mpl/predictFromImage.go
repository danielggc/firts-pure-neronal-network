package mpl

import (
	"firtsNeural/operations"
	"image"
)

func (net *Network) PredictFromImage(img image.Image) int {
	var max float64 = 0
	var highest int = 0
	dataImg := operations.GetDataFromImage(img)
	result := net.Predict(dataImg)
	for i := 0; i < net.outputs; i++ {
		if result.At(i, 0) > max {
			max = result.At(i, 0)
			highest = i
		}
		println(result.At(i, 0))
	}
	return (highest)
}
