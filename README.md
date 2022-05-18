# Video Motion Magnification

## Installation
Create a new virtual environment, activate it and use command:
  - pip install -r requirements.txt

I used Visual Studio Code and created and activated my environment using these commands:
  - py -3 -m venv .venv
  - Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process   *in my case this command is necessary any time I want to activate env*
  - .venv\scripts\activate

## Usage

#### linear_based.py  
This script was written/edited by using these repos as a source ([Repo 1](https://github.com/brycedrennan/eulerian-magnification), [Repo 2](https://github.com/flyingzhao/PyEVM)). Corresponding [Paper](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)

Specify parameters inside the main function and run the script. 

#### phase_based.py 
This script was written/edited by using this repo as a source ([Repo](https://github.com/jvgemert/pbMoMa)). Corresponding [Paper](http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf)

Specify parameters inside the main function and run the script. 

#### optical_flow.py 
In the beginning of the script we set the parameters. If we set ```save_as_video``` to ```True```, the motion vector will be drawn on the video sequence and saved as a video. Furthermore, we can set the ```fps_output_video``` to a certain number if we want to 'slow down' the video.
If we set ```save_as_video``` to ```False```, we will **not** save a video but rather see a plot for every frame which we can stop by pressing letter 'q'. This option can be usefull for quick glance on the motion vector.

In the line 56, we choose a region of interest (ROI) by defining ```square_coords```. Then, we calculate the position vectors by applying a mean value of pixels inside the ROI.
In the line 64, we define the root of the motion vector ```start_X, start_Y```. Typically in the center of the ROI.
In the line 67, we can manually amplify the length of the vector for visualization purposes (currently set to 30).

