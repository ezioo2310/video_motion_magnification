# Video Motion Magnification

## Installation
Create a new virtual environment, activate it and use command:
  - pip install -r requirements.txt

I used Visual Studio Code and created and activated my environment using these commands:
  - py -3 -m venv .venv
  - Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process    #in my case this command was necessary
  - .venv\scripts\activate

## Usage

linear_based.py  ->  This script was written/edited by using these repos as a source ([Repo 1](https://github.com/brycedrennan/eulerian-magnification), [Repo 2](https://github.com/flyingzhao/PyEVM))
                 ->  Corresponding [Paper](https://people.csail.mit.edu/mrub/papers/vidmag.pdf)


    Specify the parameters inside the main and run the code


`low` and `high` specify the corresponding frequencies. `filt` specifies which temporal filter to use (choices are 'ideal' and 'butter'). `level` specificies the number of levels in the pyramid and `amplification` the amount of amplification of the frequency band determined by `low` and `high`. `rgb` determines wheteher we use RGB images or not. `custom` specifies whether we would like to use custom video loading; we use this depending on if we would like to use cropped video sequence; if we want to use it, we have to manually set the pixel range that determines the cut image.

