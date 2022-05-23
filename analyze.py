from utility import *

#loading the video. rgb flag determines whether we use RGB or Grayscale images
path_to_video = 'video/auto_grayscale_ROI.avi' 
video, fps = load_video(path_to_video, rgb = False)

#video properties
num_frames, width, height, fps = get_video_info(path_to_video)

#extracting an image from the video
image = video[0]
show_image(image)

#ploting pyramid. Upper image is originally extracted image, lower image is when histogram equalization is applied
plot_pyramid(image, pyramid_levels = 4, pyramid = 'laplacian') #laplacian pyramid is used in linear-based approach,
                                                               #this is just a cool visualisation of layers

#choosing a region of interest
square_coords = {'x1':272-100, 'x2': 272+20, 'y1':480-100, 'y2':480+100} 
draw_roi(image, square_coords)

#FFT of the image and ROI(This is optional; if you do not want to show ROI fft, set square_coords=None). 
show_frequencies(video[:-100], fps, square_coords=square_coords)
    #NOTE: sometimes it's better to pass subarray: video[x:y]. The reason is that for some videos
    #      the brightness value drops before the video ends so the average of the pixel values looks like the inverse step function, like this: ‾‾‾‾|__
    #      so it messes up the fft. For instance, this happens for the 'auto' footage.

#Because motion magnification scripts are memory demanding, the easiest way to go around this issue
#   is to crop the video by choosing the ROI and/or to use grayscale images(rgb=False).
#Example: load_video('auto.mov',rgb=False) -> choose ROI by defining square_coords with the help of draw_roi fcn and extract the video.
#       That is how 'auto_grayscale_ROI.avi' was computed.
#extract_roi_video(video, 'video_results/cropped_video.avi', fps, square_coords)