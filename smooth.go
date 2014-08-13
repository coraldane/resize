package resize

import (
	"image"
	"runtime"
	"sync"
)

func GaussianSmooth(srcImg image.Image, sigma float64, radius int) image.Image {
	bounds := srcImg.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	scaleX, scaleY := 1.0, 1.0

	cpus := runtime.NumCPU()
	wg := sync.WaitGroup{}

	// Generic access to image.Image is slow in tight loops.
	// The optimal access has to be determined from the concrete image type.
	switch input := srcImg.(type) {
	case *image.RGBA:
		// 8-bit precision
		temp := image.NewRGBA(image.Rect(0, 0, input.Bounds().Dy(), int(width)))
		result := image.NewRGBA(image.Rect(0, 0, int(width), int(height)))
		// horizontal filter, results in transposed temporary image
		coeffs, offset, filterLength := createSmooth8(temp.Bounds().Dy(), input.Bounds().Min.X, radius, sigma, Gaussian)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(temp, i, cpus).(*image.RGBA)
			go func() {
				defer wg.Done()
				resizeRGBA(input, slice, scaleX, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()

		// horizontal filter on transposed image, result is not transposed
		coeffs, offset, filterLength = createSmooth8(result.Bounds().Dy(), temp.Bounds().Min.X, radius, sigma, Gaussian)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(result, i, cpus).(*image.RGBA)
			go func() {
				defer wg.Done()
				resizeRGBA(temp, slice, scaleY, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()
		return result
	case *image.YCbCr:
		// 8-bit precision
		// accessing the YCbCr arrays in a tight loop is slow.
		// converting the image to ycc increases performance by 2x.
		temp := newYCC(image.Rect(0, 0, input.Bounds().Dy(), int(width)), input.SubsampleRatio)
		result := newYCC(image.Rect(0, 0, int(width), int(height)), input.SubsampleRatio)

		coeffs, offset, filterLength := createSmooth8(temp.Bounds().Dy(), input.Bounds().Min.X, radius, sigma, Gaussian)
		in := ImageYCbCrToYCC(input)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(temp, i, cpus).(*ycc)
			go func() {
				defer wg.Done()
				resizeYCbCr(in, slice, scaleX, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()

		coeffs, offset, filterLength = createSmooth8(result.Bounds().Dy(), temp.Bounds().Min.X, radius, sigma, Gaussian)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(result, i, cpus).(*ycc)
			go func() {
				defer wg.Done()
				resizeYCbCr(temp, slice, scaleY, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()
		return result.YCbCr()
	case *image.RGBA64:
		// 16-bit precision
		temp := image.NewRGBA64(image.Rect(0, 0, input.Bounds().Dy(), int(width)))
		result := image.NewRGBA64(image.Rect(0, 0, int(width), int(height)))

		// horizontal filter, results in transposed temporary image
		coeffs, offset, filterLength := createSmooth16(temp.Bounds().Dy(), input.Bounds().Min.X, radius, sigma, Gaussian)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(temp, i, cpus).(*image.RGBA64)
			go func() {
				defer wg.Done()
				resizeRGBA64(input, slice, scaleX, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()

		// horizontal filter on transposed image, result is not transposed
		coeffs, offset, filterLength = createSmooth16(result.Bounds().Dy(), temp.Bounds().Min.X, radius, sigma, Gaussian)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(result, i, cpus).(*image.RGBA64)
			go func() {
				defer wg.Done()
				resizeGeneric(temp, slice, scaleY, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()
		return result
	case *image.Gray:
		// 8-bit precision
		temp := image.NewGray(image.Rect(0, 0, input.Bounds().Dy(), int(width)))
		result := image.NewGray(image.Rect(0, 0, int(width), int(height)))

		// horizontal filter, results in transposed temporary image
		coeffs, offset, filterLength := createSmooth8(temp.Bounds().Dy(), input.Bounds().Min.X, radius, sigma, Gaussian)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(temp, i, cpus).(*image.Gray)
			go func() {
				defer wg.Done()
				resizeGray(input, slice, scaleX, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()

		// horizontal filter on transposed image, result is not transposed
		coeffs, offset, filterLength = createSmooth8(result.Bounds().Dy(), temp.Bounds().Min.X, radius, sigma, Gaussian)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(result, i, cpus).(*image.Gray)
			go func() {
				defer wg.Done()
				resizeGray(temp, slice, scaleY, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()
		return result
	case *image.Gray16:
		// 16-bit precision
		temp := image.NewGray16(image.Rect(0, 0, input.Bounds().Dy(), int(width)))
		result := image.NewGray16(image.Rect(0, 0, int(width), int(height)))

		// horizontal filter, results in transposed temporary image
		coeffs, offset, filterLength := createSmooth16(temp.Bounds().Dy(), input.Bounds().Min.X, radius, sigma, Gaussian)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(temp, i, cpus).(*image.Gray16)
			go func() {
				defer wg.Done()
				resizeGray16(input, slice, scaleX, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()

		// horizontal filter on transposed image, result is not transposed
		coeffs, offset, filterLength = createSmooth16(result.Bounds().Dy(), temp.Bounds().Min.X, radius, sigma, Gaussian)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(result, i, cpus).(*image.Gray16)
			go func() {
				defer wg.Done()
				resizeGray16(temp, slice, scaleY, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()
		return result
	default:
		// 16-bit precision
		temp := image.NewRGBA64(image.Rect(0, 0, srcImg.Bounds().Dy(), int(width)))
		result := image.NewRGBA64(image.Rect(0, 0, int(width), int(height)))

		// horizontal filter, results in transposed temporary image
		coeffs, offset, filterLength := createSmooth16(temp.Bounds().Dy(), input.Bounds().Min.X, radius, sigma, Gaussian)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(temp, i, cpus).(*image.RGBA64)
			go func() {
				defer wg.Done()
				resizeGeneric(srcImg, slice, scaleX, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()

		// horizontal filter on transposed image, result is not transposed
		coeffs, offset, filterLength = createSmooth16(result.Bounds().Dy(), temp.Bounds().Min.X, radius, sigma, Gaussian)
		wg.Add(cpus)
		for i := 0; i < cpus; i++ {
			slice := makeSlice(result, i, cpus).(*image.RGBA64)
			go func() {
				defer wg.Done()
				resizeRGBA64(temp, slice, scaleY, coeffs, offset, filterLength)
			}()
		}
		wg.Wait()
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

// range [-65536,65536]
func createSmooth16(dy, minx, filterLength int, sigma float64, kernel func(float64, int) float64) ([]int32, []int, int) {
	coeffs := make([]int32, dy*filterLength)
	start := make([]int, dy)
	for y := 0; y < dy; y++ {
		interpX := float64(y) + 0.5 + float64(minx)
		start[y] = int(interpX) - filterLength/2 + 1
		for i := 0; i < filterLength; i++ {
			coeffs[y*filterLength+i] = int32(kernel(sigma, i) * 65536)
		}
	}

	return coeffs, start, filterLength
}
