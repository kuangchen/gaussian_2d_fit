from gaussian_2d_fit import Gaussian2DFit, Gaussian2DFitError
import sys
import scipy
import png

if __name__ == "__main__":
    fn = sys.argv[1]
    
    fd = open(fn, 'rb')
    format, width, height, samples, maxval = png.read_pnm_header(fd, supported=('P5', 'P6', 'P2') )
    pixels = scipy.loadtxt(fd, dtype=scipy.uint16)
    img = pixels.reshape(height,width)

    fit = Gaussian2DFit(img)
    fit.save_fit_image("fig_fit.png")
    print(fit.get_fit_param())

