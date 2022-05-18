from __future__ import division
from matplotlib.cbook import flatten
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fft, ifft, fftfreq
import copy
import scipy.fftpack as fftpack
from scipy.special import factorial


import numpy as np
import scipy.misc as sc
import scipy.signal


def visualize(coeff, normalize = True):
	M, N = coeff[1][0].shape
	Norients = len(coeff[1])
	out = np.zeros((M * 2 - coeff[-1].shape[0], Norients * N))

	currentx = 0
	currenty = 0
	for i in range(1, len(coeff[:-1])):
		for j in range(len(coeff[1])):
			tmp = coeff[i][j].real
			m,n = tmp.shape

			if normalize:
				tmp = 255 * tmp/tmp.max()

			tmp[m - 1, :] = 255
			tmp[:, n - 1] = 2555

			out[currentx : currentx + m, currenty : currenty + n] = tmp
			currenty += n
		currentx += coeff[i][0].shape[0]
		currenty = 0
	
	m,n = coeff[-1].shape
	out[currentx : currentx+m, currenty : currenty+n] = 255 * coeff[-1]/coeff[-1].max()

	out[0,:] = 255
	out[:, 0] = 255

	return out

class Steerable:
	def __init__(self, height = 5):
		"""
		height is the total height, including highpass and lowpass
		"""
		self.nbands = 4
		self.height = height
		self.isSample = True

	def buildSCFpyr(self, im):
		assert len(im.shape) == 2, 'Input image must be grayscale'

		M, N = im.shape
		log_rad, angle = self.base(M, N)
		Xrcos, Yrcos = self.rcosFn(1, -0.5)
		Yrcos = np.sqrt(Yrcos)
		YIrcos = np.sqrt(1 - Yrcos*Yrcos)

		lo0mask = self.pointOp(log_rad, YIrcos, Xrcos)
		hi0mask = self.pointOp(log_rad, Yrcos, Xrcos)

		imdft = np.fft.fftshift(np.fft.fft2(im))
		lo0dft = imdft * lo0mask

		coeff = self.buildSCFpyrlevs(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height - 1)

		hi0dft = imdft * hi0mask
		hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))

		coeff.insert(0, hi0.real)

		return coeff

	def getlist(self, coeff):
		straight = [bands for scale in coeff[1:-1] for bands in scale]
		straight = [coeff[0]] + straight + [coeff[-1]]
		return straight

	def buildSCFpyrlevs(self, lodft, log_rad, angle, Xrcos, Yrcos, ht):
		if (ht <=1):
			lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
			coeff = [lo0.real]
		
		else:
			Xrcos = Xrcos - 1

			# ==================== Orientation bandpass =======================
			himask = self.pointOp(log_rad, Yrcos, Xrcos)

			lutsize = 1024
			Xcosn = np.pi * np.array(range(-(2*lutsize+1),(lutsize+2)))/lutsize
			order = self.nbands - 1
			const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))

			alpha = (Xcosn + np.pi) % (2*np.pi) - np.pi
			Ycosn = 2*np.sqrt(const) * np.power(np.cos(Xcosn), order) * (np.abs(alpha) < np.pi/2)

			orients = []

			for b in range(self.nbands):
				anglemask = self.pointOp(angle, Ycosn, Xcosn + np.pi*b/self.nbands)
				banddft = np.power(complex(0,-1), self.nbands - 1) * lodft * anglemask * himask
				band = np.fft.ifft2(np.fft.ifftshift(banddft))
				orients.append(band)

			# ================== Subsample lowpass ============================
			dims = np.array(lodft.shape)
			
			lostart = np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)
			loend = lostart + np.ceil((dims-0.5)/2)

			lostart = lostart.astype(int)
			loend = loend.astype(int)

			log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
			angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
			lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
			YIrcos = np.abs(np.sqrt(1 - Yrcos*Yrcos))
			lomask = self.pointOp(log_rad, YIrcos, Xrcos)

			lodft = lomask * lodft

			coeff = self.buildSCFpyrlevs(lodft, log_rad, angle, Xrcos, Yrcos, ht-1)
			coeff.insert(0, orients)

		return coeff

	def reconSCFpyrLevs(self, coeff, log_rad, Xrcos, Yrcos, angle):

		if (len(coeff) == 1):
			return np.fft.fftshift(np.fft.fft2(coeff[0]))

		else:

			Xrcos = Xrcos - 1
    		
    		# ========================== Orientation residue==========================
			himask = self.pointOp(log_rad, Yrcos, Xrcos)

			lutsize = 1024
			Xcosn = np.pi * np.array(range(-(2*lutsize+1),(lutsize+2)))/lutsize
			order = self.nbands - 1
			const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
			Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

			orientdft = np.zeros(coeff[0][0].shape)

			for b in range(self.nbands):
				anglemask = self.pointOp(angle, Ycosn, Xcosn + np.pi* b/self.nbands)
				banddft = np.fft.fftshift(np.fft.fft2(coeff[0][b]))
				orientdft = orientdft + np.power(complex(0,1), order) * banddft * anglemask * himask

			# ============== Lowpass component are upsampled and convoluted ============
			dims = np.array(coeff[0][0].shape)
			
			lostart = (np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)).astype(np.int32)
			loend = lostart + np.ceil((dims-0.5)/2).astype(np.int32) 
			lostart = lostart.astype(int)
			loend = loend.astype(int)

			nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
			nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
			YIrcos = np.sqrt(np.abs(1 - Yrcos * Yrcos))
			lomask = self.pointOp(nlog_rad, YIrcos, Xrcos)

			nresdft = self.reconSCFpyrLevs(coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)

			res = np.fft.fftshift(np.fft.fft2(nresdft))

			resdft = np.zeros(dims, 'complex')
			resdft[lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask

			return resdft + orientdft

	def reconSCFpyr(self, coeff):

		if (self.nbands != len(coeff[1])):
			raise Exception("Unmatched number of orientations")

		M, N = coeff[0].shape
		log_rad, angle = self.base(M, N)

		Xrcos, Yrcos = self.rcosFn(1, -0.5)
		Yrcos = np.sqrt(Yrcos)
		YIrcos = np.sqrt(np.abs(1 - Yrcos*Yrcos))

		lo0mask = self.pointOp(log_rad, YIrcos, Xrcos)
		hi0mask = self.pointOp(log_rad, Yrcos, Xrcos)

		tempdft = self.reconSCFpyrLevs(coeff[1:], log_rad, Xrcos, Yrcos, angle)

		hidft = np.fft.fftshift(np.fft.fft2(coeff[0]))
		outdft = tempdft * lo0mask + hidft * hi0mask

		return np.fft.ifft2(np.fft.ifftshift(outdft)).real.astype(int)


	def base(self, m, n):
		
		x = np.linspace(-(m // 2)/(m / 2), (m // 2)/(m / 2) - (1 - m % 2)*2/m , num = m)
		y = np.linspace(-(n // 2)/(n / 2), (n // 2)/(n / 2) - (1 - n % 2)*2/n , num = n)

		xv, yv = np.meshgrid(y, x)

		angle = np.arctan2(yv, xv)

		rad = np.sqrt(xv**2 + yv**2)
		rad[m//2][n//2] = rad[m//2][n//2 - 1]
		log_rad = np.log2(rad)

		return log_rad, angle

	def rcosFn(self, width, position):
		N = 256
		X = np.pi * np.array(range(-N-1, 2))/2/N

		Y = np.cos(X)**2
		Y[0] = Y[1]
		Y[N+2] = Y[N+1]

		X = position + 2*width/np.pi*(X + np.pi/4)
		return X, Y

	def pointOp(self, im, Y, X):
		out = np.interp(im.flatten(), X, Y)
		return np.reshape(out, im.shape)

class SteerableNoSub(Steerable):

	def buildSCFpyrlevs(self, lodft, log_rad, angle, Xrcos, Yrcos, ht):
		if (ht <=1):
			lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
			coeff = [lo0.real]
		
		else:
			Xrcos = Xrcos - 1

			# ==================== Orientation bandpass =======================
			himask = self.pointOp(log_rad, Yrcos, Xrcos)

			lutsize = 1024
			Xcosn = np.pi * np.array(range(-(2*lutsize+1),(lutsize+2)))/lutsize
			order = self.nbands - 1
			const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))

			alpha = (Xcosn + np.pi) % (2*np.pi) - np.pi
			Ycosn = 2*np.sqrt(const) * np.power(np.cos(Xcosn), order) * (np.abs(alpha) < np.pi/2)

			orients = []

			for b in range(self.nbands):
				anglemask = self.pointOp(angle, Ycosn, Xcosn + np.pi*b/self.nbands)
				banddft = np.power(complex(0,-1), self.nbands - 1) * lodft * anglemask * himask
				band = np.fft.ifft2(np.fft.ifftshift(banddft))
				orients.append(band)

			# ================== Subsample lowpass ============================
			lostart = (0, 0)
			loend = lodft.shape

			log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
			angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
			lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
			YIrcos = np.abs(np.sqrt(1 - Yrcos*Yrcos))
			lomask = self.pointOp(log_rad, YIrcos, Xrcos)

			lodft = lomask * lodft

			coeff = self.buildSCFpyrlevs(lodft, log_rad, angle, Xrcos, Yrcos, ht-1)
			coeff.insert(0, orients)

		return coeff
#############################################################


class SlidingWindow (object):
    
    def __init__(self, size, step=1):
        self.size = size
        self.step = step
        self.memory = None
        
        assert(self.step > 0)
    
    def process(self, data_itr):
        """ 
            Generator for windows after giving it more data.
        
            Example:
            
            winsize = 2
            win = SlidingWindow(winsize)
            batches = (np.random.randint(0,9, 3) for _ in range(3))
            for w in win.process(batches):
                print '<<<', w        
        """
        for data in data_itr:
            self.update(data)
            while True:
                try:
                    out = self.next()
                    yield out
                except StopIteration:
                    break
    
    def update(self, data):
        if self.memory is None:
            self.memory = np.asarray(data)
        else:
            self.memory = np.concatenate((self.memory, data), axis=0)
        
    def next(self):
        if self.memory is not None and self.memory.shape[0] >= self.size:
            # get window
            out = self.memory[:self.size]
            
            # slide
            self.memory = self.memory[self.step:]
            
            return out
        else:
            raise StopIteration()
    
    def collect(self):
        # collect remainder of sliding windows
        out = []
        while True:
            try:
                out.append(self.next())
            except StopIteration:
                break
        return np.array(out)

class IdealFilter (object):
    """ Implements ideal_bandpassing as in EVM_MAtlab. """
    
    def __init__(self, wl=.5, wh=.75, fps=1, NFFT=None):
        """Ideal bandpass filter using FFT """

        self.fps = fps
        self.wl = wl
        self.wh = wh
        self.NFFT = NFFT
        
        if self.NFFT is not None:
            self.__set_mask()
            
    def __set_mask(self):
        self.frequencies = fftpack.fftfreq(self.NFFT, d=1.0/self.fps)    
        
        # determine what indices in Fourier transform should be set to 0
        self.mask = (np.abs(self.frequencies) < self.wl) | (np.abs(self.frequencies) > self.wh)

    def __call__(self, data, axis=0):
        if self.NFFT is None:
            self.NFFT = data.shape[0]
            self.__set_mask()            
            
        fft = fftpack.fft(data, axis=axis)        
        fft[self.mask] = 0   
        return np.real( fftpack.ifft(fft, axis=axis) )        

class IdealFilterWindowed (SlidingWindow):
    
    def __init__(self, winsize, wl=.5, wh=.75, fps=1, step=1, outfun=None):
        SlidingWindow.__init__(self, winsize, step)
        self.filter = IdealFilter(wl, wh, fps=fps, NFFT=winsize)
        self.outfun = outfun
        
    def next(self):
        out = SlidingWindow.next(self)
        out = self.filter(out)
        if self.outfun is not None:
            # apply output function, e.g. to return first (most recent) item
            out = self.outfun(out)
        return out


class IIRFilter (SlidingWindow):
    """ 
    Implements the IIR filter
           a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[nb]*x[n-nb]
                                   - a[1]*y[n-1] - ... - a[na]*y[n-na]        
    See scipy.signal.lfilter
    """

    def __init__(self, b, a):
        
        self.b = b
        self.a = a
        self.nb = len(b)
        self.na = len(a)
        
        # put parameters in right order for calculation
        #  (i.e. parameter of most recent time step last)
        self.b_ = b[::-1]
        self.a_ = a[-1:0:-1] # exclude a[0], it's used to scale output
        
        # setup sliding windows for input x and output y
        self.windowy = SlidingWindow(self.na-1)
        SlidingWindow.__init__(self, self.nb)
        
    def update(self, data):
        if self.memory is None:
            # prepend zeros
            data = np.asarray(data)
            zsize = (self.nb-1,) + data.shape[1:]
            data = np.concatenate((np.zeros(zsize), data), axis=0)
            
            # initialize output memory with zerostoo
            zsize = (self.na-1,) + data.shape[1:]
            self.windowy.update(np.zeros(zsize))
            
        SlidingWindow.update(self, data)

    def next(self):
        winx = SlidingWindow.next(self)
        winy = self.windowy.next()
        y = np.dot(self.b_, winx) - np.dot(self.a_, winy)

        self.windowy.update([y])
            
        return y / self.a[0]

        
class ButterFilter (IIRFilter):
    def __init__(self, n, freq, fps=1, btype='low'):
        freq = float(freq) / fps
        (b,a) = scipy.signal.butter(n, freq, btype)
        IIRFilter.__init__(self, b, a)


class ButterBandpassFilter (ButterFilter):
    
    def __init__(self, n, freq_low=.25, freq_high=.5, fps=1):
        ButterFilter.__init__(self, n, freq_high, fps=fps, btype='low')

        # additional low-pass
        self.lowpass = ButterFilter(n, freq_low, fps=fps, btype='low')
        
    def update(self, data):
        ButterFilter.update(self, data)
        self.lowpass.update(data)
    
    def next(self):
        out = ButterFilter.next(self)
        out_low = self.lowpass.next()
        return (out - out_low)


class Pyramid2arr:
    '''Class for converting a pyramid to/from a 1d array'''
    
    def __init__(self, steer, coeff=None):
        """
        Initialize class with sizes from pyramid coeff
        """
        self.levels = range(1, steer.height-1)
        self.bands = range(steer.nbands)
        
        self._indices = None
        if coeff is not None:
            self.init_coeff(coeff)

    def init_coeff(self, coeff):       
        shapes = [coeff[0].shape]        
        for lvl in self.levels:
            for b in self.bands:
                shapes.append( coeff[lvl][b].shape )             
        shapes.append(coeff[-1].shape)

        # compute the total sizes        
        sizes = [np.prod(shape) for shape in shapes]
        
        # precompute indices of each band
        offsets = np.cumsum([0] + sizes)
        self._indices = zip(offsets[:-1], offsets[1:], shapes)

    def p2a(self, coeff):
        """
        Convert pyramid as a 1d Array
        """
        
        if self._indices is None:
            self.init_coeff(coeff)
        
        bandArray = np.hstack([ np.ravel( coeff[lvl][b] ) for lvl in self.levels for b in self.bands ])
        bandArray = np.hstack((np.ravel(coeff[0]), bandArray, np.ravel(coeff[-1])))

        return bandArray        
        

    def reconstruct_coeff(self, coeff, new_arr):
        nlevels = self.levels
        nbands = self.bands

        first_shape = coeff[0].shape
        first_flat_len = np.prod(coeff[0].shape)
        
        first_img = new_arr[0:first_flat_len]
        first_img_reshape = first_img.reshape(first_shape)
        result = [first_img_reshape]

        current_index = first_flat_len

        for lvl in nlevels:
            to_add = []
            for b in (nbands):
                start_index = current_index 

                part_length = np.prod(coeff[lvl][b].shape)
                end_index = current_index + part_length

                curr_img = new_arr[start_index:end_index]
                curr_img_reshaped = curr_img.reshape(coeff[lvl][b].shape)
                to_add.append(curr_img_reshaped)

                current_index = end_index
                
            result.append(to_add)


        last_shape = coeff[-1].shape
        last_flat_len = np.prod(coeff[-1].shape)
        
        last_img = new_arr[current_index: (current_index+last_flat_len)]
        last_img_reshape = last_img.reshape(last_shape)
        result.append(last_img_reshape)


        # print("results shape", len(result))
        return result

    def flatten_coeff(self, coeff):

        nlevels = self.levels
        nbands = self.bands

        coeff_1d = list(coeff[0].flatten())
        for lvl in range(1, nlevels-1):
            for b in range(nbands):
                coeff_1d.extend( coeff[lvl][b].flatten() )             
        coeff_1d.extend(coeff[-1].flatten())
        coeff_1d = np.array(coeff_1d)
        return coeff_1d


    def a2p(self, bandArray):
        """
        Convert 1d array back to Pyramid
        """
        
        assert self._indices is not None, 'Initialize Pyramid2arr first with init_coeff() or p2a()'

        # create iterator that convert array to images
        it = (np.reshape(bandArray[istart:iend], size) for (istart,iend,size) in self._indices)
        import pdb; pdb.set_trace()
        coeffs = [next(it)]
        for lvl in self.levels:
            coeffs.append([next(it) for band in self.bands])
        coeffs.append(next(it))

        return coeffs


import cv2

def phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, lowFreq, highFreq):

    # initialize the steerable complex pyramid
    steer = Steerable(5)
    pyArr = Pyramid2arr(steer)

    # get vid properties
    vidReader = cv2.VideoCapture(vidFname)

    # OpenCV 3.x interface
    vidFrames = int(vidReader.get(cv2.CAP_PROP_FRAME_COUNT))    
    width = int(vidReader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidReader.get(cv2.CAP_PROP_FPS))
    func_fourcc = cv2.VideoWriter_fourcc

    if np.isnan(fps):
        fps = 30

    # video Writer
    fourcc = func_fourcc('M', 'J', 'P', 'G')
    vidWriter = cv2.VideoWriter(vidFnameOut, fourcc, int(fps), (width,height), 1)


    # how many frames
    nrFrames = min(vidFrames, maxFrames)

    # read video
    #print steer.height, steer.nbands

    # setup temporal filter
    filter = IdealFilterWindowed(windowSize, lowFreq, highFreq, fps=fps, outfun=lambda x: x[0])
    #filter = ButterBandpassFilter(1, lowFreq, highFreq, fps=fps)


    for frameNr in range( nrFrames + windowSize ):
        print(f"The frame number {frameNr} is being processed")
        # sys.stdout.flush() 

        if frameNr < nrFrames:
            # read frame
            _, im = vidReader.read()
               
            if im is None:
                # if unexpected, quit
                break

            # convert to gray image
            if len(im.shape) > 2:
                grayIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            else:
                # already a grayscale image?
                grayIm = im

            # get coeffs for pyramid
            coeff = steer.buildSCFpyr(grayIm)
            
            # add image pyramid to video array
            # NOTE: on first frame, this will init rotating array to store the pyramid coeffs                 
            arr = pyArr.p2a(coeff)
           

            phases = np.angle(arr)

            # add to temporal filter
            filter.update([phases])

            # try to get filtered output to continue            
            try:
                filteredPhases = filter.next()
            except StopIteration:
                continue

            
            # motion magnification
            magnifiedPhases = (phases - filteredPhases) + filteredPhases*factor
            
            # create new array
            newArr = np.abs(arr) * np.exp(magnifiedPhases * 1j)

            # create pyramid coeffs     
            newCoeff = pyArr.reconstruct_coeff(coeff, newArr)
            
            # reconstruct pyramid
            out = steer.reconSCFpyr(newCoeff)

            # clip values out of range
            out[out>255] = 255
            out[out<0] = 0
            
            # make a RGB image
            rgbIm = np.empty( (out.shape[0], out.shape[1], 3 ) )
            rgbIm[:,:,0] = out
            rgbIm[:,:,1] = out
            rgbIm[:,:,2] = out
            
            #write to disk
            res = cv2.convertScaleAbs(rgbIm)
            vidWriter.write(res)

    # free the video reader/writer
    vidReader.release()
    vidWriter.release() 


if __name__ == "__main__":

    #input video path
    vidFname = 'video_results/auto_original.avi'
    # maximum nr of frames to process
    maxFrames = 1000
    # the size of the sliding window
    windowSize = 30
    # the magnifaction factor
    factor = 20
    # low ideal filter
    lowFreq = 0.5
    # high ideal filter
    highFreq = 1.5
    # output video path
    vidFnameOut = './video_results/baby_phase_based_' + f'{lowFreq}-{highFreq}Hz.avi'

    phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, lowFreq, highFreq)