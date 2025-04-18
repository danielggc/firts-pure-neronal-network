package mpl

import (
	"firtsNeural/operations"

	"gonum.org/v1/gonum/mat"
)

type Network struct {
	intputs        int
	hiddens        int
	outputs        int
	hiddensWeights *mat.Dense
	outputsWeights *mat.Dense
	learningRate   float64
}

func CreateNetwork(intputs, hiddens, outputs int, learningRate float64) Network {
	result := Network{
		intputs:      intputs,
		hiddens:      hiddens,
		outputs:      outputs,
		learningRate: learningRate,
	}

	result.hiddensWeights = mat.NewDense(hiddens, intputs, operations.RamdomArray(hiddens*intputs, intputs))
	result.outputsWeights = mat.NewDense(outputs, hiddens, operations.RamdomArray(hiddens*outputs, intputs))
	return result

}

func (net *Network) Training(image []float64, targetImage []float64) {
	input := mat.NewDense(len(image), 1, image)
	hiddensInputs := operations.ProductMatrix(net.hiddensWeights, input)
	hiddensOutputs := operations.ApplyFuntionToMatrix(operations.Sigmoid, hiddensInputs)
	finalInputs := operations.ProductMatrix(net.outputsWeights, hiddensOutputs)
	finalOutputs := operations.ApplyFuntionToMatrix(operations.Sigmoid, finalInputs)

	targetM := mat.NewDense(net.outputs, 1, targetImage)
	outputErrors := operations.SubMatrix(targetM, finalOutputs)
	hiddensErrors := operations.ProductMatrix(net.outputsWeights.T(), outputErrors)

	net.outputsWeights = operations.AddMatrix(
		net.outputsWeights,
		operations.AddScalar(
			net.learningRate,
			operations.ProductMatrix(
				operations.Multiply(
					outputErrors,
					operations.SigmoidPrime(finalOutputs),
				),
				hiddensOutputs.T(),
			),
		),
	).(*mat.Dense)

	net.hiddensWeights = operations.AddMatrix(
		net.hiddensWeights,
		operations.AddScalar(
			net.learningRate,
			operations.ProductMatrix(
				operations.Multiply(
					hiddensErrors,
					operations.SigmoidPrime(hiddensOutputs),
				),
				input.T(),
			),
		),
	).(*mat.Dense)

}

func (net Network) Predict(image []float64) mat.Matrix {
	input := mat.NewDense(len(image), 1, image)
	hiddensInputs := operations.ProductMatrix(net.hiddensWeights, input)
	hiddensOutputs := operations.ApplyFuntionToMatrix(operations.Sigmoid, hiddensInputs)
	finalInputs := operations.ProductMatrix(net.outputsWeights, hiddensOutputs)
	finalOutputs := operations.ApplyFuntionToMatrix(operations.Sigmoid, finalInputs)

	return finalOutputs
}
