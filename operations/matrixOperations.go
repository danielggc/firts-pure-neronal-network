package operations

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

func ProductMatrix(fa, fb mat.Matrix) mat.Matrix {
	sizeRow, _ := fa.Dims()
	_, sizeColumn := fb.Dims()
	newMatrix := mat.NewDense(sizeRow, sizeColumn, nil)

	newMatrix.Mul(fa, fb)

	return newMatrix

}

func ApplyFuntionToMatrix(
	fn func(fa, fb int, fc float64) float64,
	matrix mat.Matrix) mat.Matrix {
	r, c := matrix.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(fn, matrix)
	return result
}

func MulByScalar(ma mat.Matrix, sc float64) mat.Matrix {
	r, c := ma.Dims()
	result := mat.NewDense(r, c, nil)
	result.Scale(sc, ma)
	return result
}

func AddMatrix(fa, fb mat.Matrix) mat.Matrix {
	r, c := fa.Dims()
	result := mat.NewDense(r, c, nil)
	result.Add(fa, fb)
	return result
}

func SubMatrix(fa, fb mat.Matrix) mat.Matrix {
	r, c := fa.Dims()
	result := mat.NewDense(r, c, nil)
	result.Sub(fa, fb)
	return result
}

func AddScalar(fb float64, fa mat.Matrix) mat.Matrix {
	r, c := fa.Dims()
	fbM := make([]float64, r*c)
	result := mat.NewDense(r, c, fbM)
	return AddMatrix(result, fa)
}

func RamdomArray(size int, variance int) []float64 {
	result := make([]float64, size)
	dist := distuv.Uniform{
		Min: (-1 / math.Sqrt(float64(variance))),
		Max: (1 / math.Sqrt(float64(variance))),
	}
	for i := 0; i < size; i++ {
		result[i] = dist.Rand()
	}
	return result

}

func Multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func Sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z)) // σ(z) = 1 / (1 + e^(-z))
}

func SigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return Multiply(m, SubMatrix(ones, m)) // m * (1 - m)
}
