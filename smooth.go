package resize

import (
	"image"
	"runtime"
	"sync"
)

func GuassianSmooth(srcImg image.Image, sigma float64, radius int) image.Image {
	bounds := srcImg.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	cpus := runtime.NumCPU()
	wg := sync.WaitGroup{}

	// Generic access to image.Image is slow in tight loops.
	// The optimal access has to be determined from the concrete image type.
	switch input := srcImg.(type) {
	case *image.RGBA:
		// 8-bit precision
		// temp := image.NewRGBA(image.Rect(0, 0, input.Bounds().Dy(), int(width)))
		result := image.NewRGBA(image.Rect(0, 0, int(width), int(height)))
		//TODO
		return result
	case *image.YCbCr:
		// 8-bit precision
		// accessing the YCbCr arrays in a tight loop is slow.
		// converting the image to ycc increases performance by 2x.
		temp := newYCC(image.Rect(0, 0, input.Bounds().Dy(), int(width)), input.SubsampleRatio)
		result := newYCC(image.Rect(0, 0, int(width), int(height)), input.SubsampleRatio)

		coeffs, offset, filterLength := createSmooth8(temp.Bounds().Dy(), input.Bounds().Min.X, radius, sigma, Guassian)
		in := ImageYCbCrToYCC(input)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(temp, i, cpus).(*ycc)
			go func() {
				defer wg.Done()
				resizeYCbCr(in, slice, 1.0, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()

		coeffs, offset, filterLength = createSmooth8(result.Bounds().Dy(), temp.Bounds().Min.X, radius, sigma, Guassian)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(result, i, cpus).(*ycc)
			go func() {
				defer wg.Done()
				resizeYCbCr(temp, slice, 1.0, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()
		return result.YCbCr()
	default:
		// 16-bit precision
		// temp := image.NewRGBA64(image.Rect(0, 0, img.Bounds().Dy(), int(width)))
		result := image.NewRGBA64(image.Rect(0, 0, int(width), int(height)))
		//TODO
		return result
	}
}

// range [-256,256]
func createSmooth8(dy, minx, filterLength int, sigma float64, kernel func(float64, int) float64) ([]int16, []int, int) {
	coeffs := make([]int16, dy*filterLength)
	start := make([]int, dy)
	for y := 0; y < dy; y++ {
		interpX := float64(y) + 0.5 + float64(minx)
		start[y] = int(interpX) - filterLength/2 + 1
		for i := 0; i < filterLength; i++ {
			coeffs[y*filterLength+i] = int16(kernel(sigma, i) * 256)
		}
	}

	return coeffs, start, filterLength
}
