package operations

import (
	"fmt"
	"image"
	"image/png"
	"log"
	"os"

	"github.com/qeesung/image2ascii/convert"
)

func GetDataFromImage(img image.Image) []float64 {

	bounds := img.Bounds()
	gray := image.NewGray(bounds)
	for x := bounds.Min.X; x < bounds.Max.X; x++ {
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			bit := img.At(x, y)
			gray.Set(x, y, bit)
		}
	}
	pixels := make([]float64, len(gray.Pix))

	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.999) + 0.001
	}
	return pixels
}

func GetImage(filePath string) image.Image {
	imageFl, err := os.Open(filePath)
	defer imageFl.Close()
	if err != nil {
		log.Fatal("we cant read the image ")
	}
	img, err := png.Decode(imageFl)
	if err != nil {
		log.Fatal("we cant decode the image ")
	}
	return img

}

func PrintImage(filePath string) {
	convertOptions := convert.DefaultOptions
	convertOptions.FixedWidth = 40
	convertOptions.FixedHeight = 20
	converter := convert.NewImageConverter()

	result := converter.ImageFile2ASCIIString(filePath, &convertOptions)

	fmt.Println(result)
}
