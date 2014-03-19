import scipy
import Image
import ImageDraw
import scipy.optimize as optimize

class Gaussian2DFitError(Exception):
    def __init__(self, msg):
        self.message = msg

class Gaussian2DFit:
    def __init__(self, image):
        """ 
        Class constructor
        ===================
        image : a scipy.ndarray instance of 2d
        """ 
        
        if not (isinstance(image, scipy.ndarray) and image.ndim == 2):
            raise Gaussian2DFitError("image is not scipy.ndarray of 2 dims")

        self._orig_image = image
        self._shape = self._orig_image.shape
        self._p, self._cov = self._calc_fit_param()

    def get_fit_param(self):
        """
        Get fit parameter
        ====================
        Return: p, cov
        Note: p = [ A, B, x0, y0, sx, sy, theta ]
              cov is the covariance matrix
        """

        return self._p, self._cov

    def save_fit_image(self, fname):
        """ 
        Save image with border
        ====================
        fname: name of the file to be saved
        """

        A, B, x0, y0, sx, sy, theta = self._p
        sx, sy = scipy.fabs([sx, sy])

        # First draw an upright ellipse
        shape = (self._shape[1], self._shape[0])

        border = Image.new('L', shape, "black")
        draw = ImageDraw.Draw(border)
        bbox = ( y0-sy/2, x0-sx/2, y0+sy/2, x0+sx/2 )
        draw.ellipse(bbox, outline='white')

        # Then rotate the ellipse around (x0, y0) with angle theta
        c, s = scipy.cos(theta), -scipy.sin(theta)
        
        matrix = ( c, -s, y0 * (1-c) + x0 * s,
                   s,  c, x0 * (1-c) - y0 * s )

        border = border.transform(shape, Image.AFFINE, matrix)

        # Finally paste 
        rescaled = (255.0 / self._orig_image.max() * self._orig_image).astype(scipy.uint8)
        new_img = Image.fromarray(rescaled)
        new_img.paste(border, new_img)
        new_img.save(fname)
        

    def _calc_fit_param(self):

        def gaussian(A, B, x0, y0, sx, sy, theta):
            """
            Returns a gaussian function centered at (x0, y0) with 
            stdev = (sx, sy), and rotated at angle theta
            """
            c, s = scipy.cos(theta), scipy.sin(theta)

            def f(x, y):
                xx = (c * (x-x0) - s * (y-y0)) / sx
                yy = (s * (x-x0) + c * (y-y0)) / sy
                return A*scipy.exp( -(xx**2+yy**2)/2 ) + B
            return f
    
        def error(p):
            X, Y = scipy.indices(self._shape)
            g = gaussian(*p)

            return (g(X, Y) - self._orig_image).ravel()

        estimate = self._estimate_fit_param()
        p, cov, infodict, mesg, ier = optimize.leastsq(error, estimate, 
                                                       full_output=True)

        success = ier in [1,2,3,4]
                
        if not success:
            raise Gaussian2dFitError

        else:
            # Make sure sx and sy are positive
            p[4], p[5] = scipy.fabs([p[4], p[5]])
            
            dof = self._orig_image.size - len(p)
            s_sq = (infodict['fvec']**2).sum() / dof
            cov = cov * s_sq

        return p, cov

    def _estimate_fit_param(self):
        B = self._orig_image.min()
        w = self._orig_image - B
        A = w.max() 

        X, Y = scipy.indices(self._shape)

        x0 = scipy.average(X, None, w)
        y0 = scipy.average(Y, None, w)

        col = w[:, int(y0)]
        var_x = scipy.average((scipy.arange(col.size) - y0)**2, None, col)

        row = w[int(x0), :]
        var_y = scipy.average((scipy.arange(row.size) - x0)**2, None, row)
    
        return A, B, x0, y0, var_x**0.5, var_y**0.5, 0
