# CRAFT
This repository contains the authors' official public-facing implementation of the Conversational Recurrent Architecture for ForecasTing (CRAFT) neural model, as introduced in the EMNLP 2019 paper "Trouble on the Horizon: Forecasting the Derailment of Online Conversations as they Develop".

## Prerequisites
CRAFT was originally written in Python 3.5 and PyTorch 0.9, but has been tested for reproducibility up to Python 3.8 and PyTorch 1.5.

The two core requirements are PyTorch (which CRAFT is implemented in) and ConvoKit (which this implementation uses for conversational data representation and processing). We only officially support running this code via Anaconda, under which both requirements can be installed as follows:
  - PyTorch: `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
  - ConvoKit: `pip install convokit`

Additionally, NLTK is used for text preprocessing, and Pandas + Matplotlib are used for visualization (in the demo notebook). Install these additional dependencies in one line:
  - `pip install nltk pandas matplotlib`

## Configuration
Variables for configuring both file I/O (namely input and output file locations) and CRAFT network parameters are all consolidated under model/config.py. Please consult the comments in that file for descriptions of the available settings and what they do. Note that the code can be run "as-is" without editing any of the settings, if you want to use the same parameters and data from the CRAFT paper. So, you only need to edit settings.py if (a) you want to customize file paths, or (b) you want to experiment with different network parameters.

## Running the code

### Pre-training
The first step in getting a functioning CRAFT model up and running is the pre-training phase. As described in the paper, this consists of using large amounts of unlabeled data to train the encoder layers of CRAFT using a generative training objective. Due to the large amounts of data involved, this is the step that will take the longest amount of time; under the default settings, you should expect this to take at least 24 hours and possibly more depending on the power of your GPU.

Pre-training can be run by executing the following command:
```
python model/train_generative_model.py
```
No command line arguments are needed or taken, as all configuration is read from config.py. As previously stated, you do not need to edit config.py if you simply want to run the standard demo, as all configuration settings have been set to correctly read the input files bundled in this repository and output to a new subdirectory at the top level of the repo directory.

### Fine-tuning and inference
TODO
