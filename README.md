# Video Motion Magnification

## Installation
Create a new virtual environment, activate it and use the command:
  - pip install -r requirements.txt

I used Visual Studio Code in Windows and created & activated my environment using these commands:
  - py -3 -m venv .venv
  - Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process   (*in my case this command is necessary any time I want to activate env*)
  - .venv\scripts\activate

## Usage

#### linear_based.py  
This script was written/edited by using these repos as the source ([Repo 1](https://github.com/brycedrennan/eulerian-magnification), [Repo 2](https://github.com/flyingzhao/PyEVM)).

Corresponding [Paper](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)

Specify parameters inside the main function and run the script. 

#### phase_based.py 
This script was written/edited by using this repo as the source ([Repo](https://github.com/jvgemert/pbMoMa)). 

Corresponding [Paper](http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf)

Specify parameters inside the main function and run the script. 

#### optical_flow.py 
In the beginning of the script we set the parameters. If we set ```save_as_video``` to ```True```, the motion vector will be drawn on the video sequence and saved as a video. Furthermore, we can set the ```fps_output_video``` to a certain number if we want to 'slow down' the video.
If we set ```save_as_video``` to ```False```, we will **not** save a video but rather see a plot for every frame which we can stop by pressing letter 'q'. This option can be usefull for quick glance on the motion vector.

In the line 56, we choose a region of interest (ROI) by defining ```square_coords```. Then, we calculate the position vectors by applying a mean over flow values of pixels inside the ROI.
In the line 64, we define the root of the motion vector ```start_X, start_Y```. Typically in the center of the ROI.
In the line 67, we can manually amplify the length of the vector for visualization purposes (currently set to 30).

#### learning_based
The folder was copied from this repo ([Repo](https://github.com/cgst/motion-magnification)).

Corresponding [Paper](https://arxiv.org/pdf/1804.02684.pdf)

This script can be used for training also but there are pretrained models which can be used for testing right away.

Examples of running the code:
```
# Get CLI help.
python main.py -h

# Amplify video 5x.
python main.py amplify data/models/20191204-b4-r0.1-lr0.0001-05.pt data/examples/baby.mp4 --amplification=5

# Amplify 7x and render side-by-side comparison video. I mostly used this command.
python main.py demo data/models/20191204-b4-r0.1-lr0.0001-05.pt data/examples/baby.mp4 baby-demo.mp4 7

# Make video collage with input, 2x and 4x amplified side by side.
python main.py demo data/models/20191204-b4-r0.1-lr0.0001-05.pt data/examples/baby.mp4 baby-demo.mp4 2 4
```
NOTE: ```python main.py ...``` shoud be called from the ./learning_based directory or just run ```python ./learning_based/main.py ...``` from the root directory.

If you encounter an error: ``` AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor' ```, do the following:
Go inside the .venv\Lib\site-packages\torch\nn\modules\upsampling.py and modify the forward function to
```
def forward(self, input: Tensor) -> Tensor:
    return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
                         #recompute_scale_factor=self.recompute_scale_factor
                         )
```

#### analyze.py
This script can be used both for pre-processing/analyzing and post-processing/analyzing.
All functions from utility.py are imported and used in step-by-step fashion.

*Preprocessing*:
Since motion magnification scripts are memory demanding, the best use of this script is to load the video, covert it to grayscale, choose Region of Interest(ROI) and 
extract the ROI footage.

*Postprocessing*:
Plot FFT and amplitude of pixel values. Additionally, ROI can be chosen to specifically show this plots for that region.

#### utility.py
Contains useful functions that can be used for preprocessing, analyzing, plotting, drawing and testing.

## Issues and constraints
These motion-magnification scripts are 'optimized' for about 16 Gb of RAM memory. If we face memory problems, the best practice is to convert videos to grayscale and to crop it using analyze.py. I would say that the solution for this problem would be to code in C++ because we have more control over memory usage but in the end, if we need to store a big array of float numbers, we are always going to end up with memory issues.. 