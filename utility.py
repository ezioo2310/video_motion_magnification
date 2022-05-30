# TO BE ADDED
import cv2
import numpy as np
import scipy.fftpack
import os
import scipy
import scipy.signal as signal
import matplotlib.pyplot as pyplot
from torch import square

### PYRAMIDS
def create_gaussian_image_pyramid(image, pyramid_levels):
    height, width = image.shape[0:2]
    #we must have the rounded values in order to reconstruct the pyramid
    # assert height % 2**pyramid_levels == 0
    # assert width % 2**pyramid_levels == 0
    if  height % 2**pyramid_levels != 0 or width % 2**pyramid_levels != 0:
        raise Exception('Height and width MUST be divisible by: 2 to the power of pyramid_levels!!')
    gauss_copy = np.ndarray(shape=image.shape, dtype="float")
    gauss_copy[:] = image
    img_pyramid = [gauss_copy]
    for pyramid_level in range(1, pyramid_levels):
        gauss_copy = cv2.pyrDown(gauss_copy)
        img_pyramid.append(gauss_copy)

    return img_pyramid


def create_laplacian_image_pyramid(image, pyramid_levels):
    gauss_pyramid = create_gaussian_image_pyramid(image, pyramid_levels)
    laplacian_pyramid = []
    for i in range(pyramid_levels - 1):
        laplacian_pyramid.append((gauss_pyramid[i] - cv2.pyrUp(gauss_pyramid[i + 1])) + 0)

    laplacian_pyramid.append(gauss_pyramid[-1])
    return laplacian_pyramid


def create_gaussian_video_pyramid(video, pyramid_levels):
    return _create_pyramid(video, pyramid_levels, create_gaussian_image_pyramid)


def create_laplacian_video_pyramid(video, pyramid_levels):
    return _create_pyramid(video, pyramid_levels, create_laplacian_image_pyramid)


def _create_pyramid(video, pyramid_levels, pyramid_fn):
    vid_pyramid = []
    # frame_count, height, width, colors = video.shape
    for frame_number, frame in enumerate(video):
        frame_pyramid = pyramid_fn(frame, pyramid_levels)

        for pyramid_level, pyramid_sub_frame in enumerate(frame_pyramid):
            if frame_number == 0:
                vid_pyramid.append(
                    np.zeros((video.shape[0], pyramid_sub_frame.shape[0], pyramid_sub_frame.shape[1], 3),
                                dtype="float"))

            vid_pyramid[pyramid_level][frame_number] = pyramid_sub_frame

    return vid_pyramid

def plot_pyramid(image, pyramid_levels, pyramid = 'laplacian'):
    """
    Plotting the pyramid of the image. Since the values of the pixels are low when making a pyramid,
    histogram equalization is applied to make it more visable on top of the original image.
    """
    if isinstance(image[0][0].item() if len(image.shape)==2 else image[0][0][0].item(),int):
        image = image/255.0
    if pyramid == 'laplacian':
        pyr = create_laplacian_image_pyramid(image, pyramid_levels)
    elif pyramid == 'gaussian':
        pyr = create_gaussian_image_pyramid(image, pyramid_levels)
    else:
        raise Exception('Only laplacian and gaussian pyramids can be computed')
    charts_x = 1
    charts_y = pyramid_levels
    pyplot.figure(f'{pyramid} pyramid',figsize=(20, 10))
    
    for i in range(pyramid_levels):
        pyplot.subplot(charts_x, charts_y, i+1)
        pyplot.xlabel(f"Pyramid Level {i+1}")
        pyr[i] = pyr[i] - np.min(pyr[i])
        if len(pyr[0].shape)==3:
            img = cv2.cvtColor(pyr[i].astype('float32'), cv2.COLOR_BGR2GRAY)
        else:
            img = pyr[i].astype('float32')
        img *=255
        img = img.astype(np.uint8)
        equ = cv2.equalizeHist(img)
        res = np.vstack((img,equ))
        pyplot.imshow(res,cmap='gray', vmin=0, vmax=255)
    pyplot.show()


#### LOADING THE VIDEO
def uint8_to_float(img):
    result = np.ndarray(shape=img.shape, dtype='float')
    result[:] = img * (1. / 255)
    return result

def float_to_uint8(img):
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = img * 255
    return result

def float_to_int8(img):
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = (img * 255) - 127
    return result

def load_video(video_filename, rgb = True):
    """Load a video into a numpy array"""
    video_filename = str(video_filename)
    print("Loading " + video_filename)
    if not os.path.isfile(video_filename):
        raise Exception("File Not Found: %s" % video_filename)
    # noinspection PyArgumentList
    capture = cv2.VideoCapture(video_filename)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = get_capture_dimensions(capture)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    x = 0
    vid_frames = np.zeros((frame_count, height, width, 3) if rgb else (frame_count, height, width), dtype='uint8')
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        vid_frames[x] = frame if rgb else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x += 1
    capture.release()
    return vid_frames, fps

def load_video_float(video_filename, rgb = True):
    vid_data, fps = load_video(video_filename, rgb)
    return uint8_to_float(vid_data), fps

def save_video(video_tensor, name='out.avi', fps=30, RGB=True):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [height,width]=video_tensor[0].shape[0:2]
    num = 0 if RGB is False else 1 
    writer = cv2.VideoWriter(name, fourcc, fps, (width, height), num)
    for i in range(0,video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()

###Choosing region of interest and plotting stuff
def show_image(image):
    cv2.imshow('The first frame',image)
    cv2.waitKey()

def show_image_nb(image):
    pyplot.figure(figsize=(20, 10))
    if len(image.shape)==2:
        pyplot.imshow(image,cmap='gray', vmin=0, vmax=255)
    else:
        pyplot.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def draw_roi(image, square_coords):
    start_point = (square_coords['y1'],square_coords['x1'])
    end_point = (square_coords['y2'],square_coords['x2'])
    image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 2)
    cv2.imshow('Region of Interest',image)
    cv2.waitKey()

def draw_roi_nb(image, square_coords):
    start_point = (square_coords['y1'],square_coords['x1'])
    end_point = (square_coords['y2'],square_coords['x2'])
    image_drawn = cv2.rectangle(image.copy(), start_point, end_point, (255, 0, 0), 2)
    show_image_nb(image_drawn)

def show_optical_flow_roi(video_input, base_point, square_coords, amplitude):
    if os.path.exists(video_input) == False:
        raise Exception('The path to the video does not exist!')
    cap = cv2.VideoCapture(video_input)
    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    position_vectors = {'X': [], 'Y': []}

    while(cap.isOpened()):
        # READING IMAGE BY IMAGE
        ret, frame = cap.read()
        if ret is False:
                break
        # CONVERTING TO GRAYSCALE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CALCULATING FLOW
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        x1, x2, y1, y2 = square_coords['x1'], square_coords['x2'],square_coords['y1'],square_coords['y2']
        position_vectors['X'].append(np.mean(flow[x1:x2,y1:y2, 0]))
        position_vectors['Y'].append(np.mean(flow[x1:x2,y1:y2, 1]))

        # DRAWING AN ARROW
        start_X, start_Y = base_point
        start_point = base_point
        end_point = (start_X + int(np.sum(position_vectors['X'][-1:])*amplitude), start_Y + int(np.sum(position_vectors['Y'][-1:])*amplitude))
        color = (0, 255, 0)
        thickness = 2
        image = cv2.arrowedLine(frame, start_point, end_point, color, thickness)

        prev_gray = gray

        # PRESS 'Q' TO EXIT
        cv2.imshow('arrow', image) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def show_frequencies(vid_data, fps, square_coords = None):
    """
    Graph the average value of the video as well as the frequency strength.
    A region of interest can be choosen too with the variable square_coords.
    """
    averages = []
    averages_square = []

    for x in range(1, vid_data.shape[0] - 1):
        averages.append(vid_data[x, :, :, :].mean() if len(vid_data[0].shape)==3 else vid_data[x, :, :].mean())
        if square_coords is not None:
            averages_square.append(vid_data[x,
                square_coords['x1']:square_coords['x2'],
                square_coords['y1']:square_coords['y2'],
                :].mean() if len(vid_data[0].shape)==3 
                
                else vid_data[x,
                square_coords['x1']:square_coords['x2'],
                square_coords['y1']:square_coords['y2']].mean())
                
    averages = averages
    averages = averages - min(averages)
    if square_coords is not None:
        averages_square = averages_square
        averages_square = averages_square - min(averages_square)

    charts_x = 1
    charts_y = 4 if square_coords else 2
    pyplot.figure(figsize=(20, 10))
    pyplot.subplots_adjust(hspace=.7)

    pyplot.subplot(charts_y, charts_x, 1)
    pyplot.title("Pixel Average of the image")
    pyplot.xlabel("Time [frame]")
    pyplot.ylabel("Brightness")
    pyplot.plot(averages)

    pyplot.subplot(charts_y, charts_x, 2)
    pyplot.title("FFT of the image")
    pyplot.xlabel("Freq (Hz)")

    freqs = scipy.fftpack.fftfreq(len(averages), d=1.0 / fps)
    fft = abs(scipy.fftpack.fft(averages))
    idx = np.argsort(freqs)

    freqs = freqs[idx]
    fft = fft[idx]

    freqs = freqs[len(freqs) // 2 + 1:]
    fft = fft[len(fft) // 2 + 1:]
    pyplot.plot(freqs, abs(fft))

    if square_coords is not None:
        pyplot.subplot(charts_y, charts_x, 3)
        pyplot.title("Pixel Average of the selected region of interest")
        pyplot.xlabel("Time")
        pyplot.ylabel("Brightness")
        pyplot.plot(averages_square)

        freqs = scipy.fftpack.fftfreq(len(averages_square), d=1.0 / fps)
        fft = abs(scipy.fftpack.fft(averages_square))
        idx = np.argsort(freqs)

        pyplot.subplot(charts_y, charts_x, 4)
        pyplot.title("FFT of the selected region of interest")
        pyplot.xlabel("Freq (Hz)")
        freqs = freqs[idx]
        fft = fft[idx]

        freqs = freqs[len(freqs) // 2 + 1:]
        fft = fft[len(fft) // 2 + 1:]
        pyplot.plot(freqs, abs(fft))
    pyplot.show()

def extract_roi_video(video, name, fps, square_coords):
    x1, x2, y1, y2 = square_coords['x1'], square_coords['x2'],square_coords['y1'],square_coords['y2']
    height = x2 - x1
    width = y2 - y1
    video_new = np.zeros((video.shape[0], height, width),dtype='uint8') if len(video[0].shape)==2 else np.zeros((video.shape[0], height, width, 3), dtype='uint8')
    for i in range(video.shape[0]):
        video_new[i] = video[i,x1:x2,y1:y2] if len(video[0].shape)==2 else video[i,x1:x2,y1:y2,:]
    save_video(video_new, name=name, fps=fps, RGB=False if len(video[0].shape)==2 else True)

def plot_butter_filter(fps, freq_min, freq_max, order):
    omega = 0.5 * fps
    low = freq_min / omega
    high = freq_max / omega
    b, a = signal.butter(order, [low, high], btype='band')
    w, h = signal.freqz(b, a, fs=fps)
    pyplot.plot(w, abs(h), label="order = %d" % order)
    
def plot_orders_butter(fps, freq_min, freq_max):
    """
    Plotting different orders of Butterworth filter
    """
    for order in [3,5,7]:
        plot_butter_filter(fps, freq_min, freq_max, order)
    pyplot.xlabel('Frequency (Hz)')
    pyplot.ylabel('Gain')
    pyplot.grid(True)
    pyplot.legend(loc='best')    
    pyplot.show()


def get_video_info(vidFname):
    vidReader = cv2.VideoCapture(vidFname)

    vidFrames = int(vidReader.get(cv2.CAP_PROP_FRAME_COUNT))    
    width = int(vidReader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidReader.get(cv2.CAP_PROP_FPS))

    print(f'The number of frames is: {vidFrames}')
    print(f'The resolution of the video is: {height}x{width}')
    print(f'Frames per second: {fps}')

    return vidFrames, width, height, fps


def get_capture_dimensions(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height



### OPTICAL FLOW FUNCTIONS 

## Approach 1: COLOR

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def show_optical_flow_color(video_input, clip_flow=None, convert_to_bgr=False):
    if os.path.exists(video_input) == False:
        raise Exception('The path to the video does not exist!')
    cap = cv2.VideoCapture(video_input)
    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    while(cap.isOpened()):
        # READING IMAGE BY IMAGE
        ret, frame = cap.read()
        if ret is False:
                break
        # CONVERTING TO GRAYSCALE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CALCULATING FLOW
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        image = flow_to_color(flow, clip_flow=clip_flow, convert_to_bgr=convert_to_bgr)

        prev_gray = gray

        # PRESS 'Q' TO EXIT
        cv2.imshow('arrow', image) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


## Approach 2: GRID
def draw_flow(im, flow, step=50, amp=2):
    h,w = im.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype('int')
    fx, fy = flow[y,x].T

    #create line endpoints
    lines = np.vstack([x,y,x+fx*amp,y+fy*amp]).T.reshape(-1,2,2)
    lines = np.int32(lines)

    #create image and draw
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (x1,y1), (x2,y2) in lines:
        cv2.arrowedLine(vis, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.circle(vis,(x1,y1),1,(0,255,0),-1)
    return vis
    

def show_optical_flow_grid(video_input, step=50, amp=2):
    if os.path.exists(video_input) == False:
        raise Exception('The path to the video does not exist!')
    cap = cv2.VideoCapture(video_input)
    ret, im = cap.read()
    prev_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    while True:
        ret, im = cap.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        #compute flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3,15,3,5,1.2,0)
        prev_gray = gray

        #plot the flow
        cv2.imshow('Optical flow',draw_flow(gray, flow, step=step, amp=amp))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
