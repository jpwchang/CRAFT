# CRAFT
This repository contains the authors' official public-facing implementation of the Conversational Recurrent Architecture for ForecasTing (CRAFT) neural model, as introduced in the EMNLP 2019 paper "Trouble on the Horizon: Forecasting the Derailment of Online Conversations as they Develop".

## Prerequisites
CRAFT was originally written in Python 3.5 and PyTorch 0.9, but has been tested for reproducibility up to Python 3.8 and PyTorch 1.5.

The two core requirements are PyTorch (which CRAFT is implemented in) and ConvoKit (which this implementation uses for conversational data representation and processing). We only officially support running this code via Anaconda, under which both requirements can be installed as follows:
  - PyTorch: `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
  - ConvoKit: `pip install convokit`

Additionally, NLTK is used for text preprocessing, and Pandas + Matplotlib are used for visualization (in the demo notebook). Install these additional dependencies in one line:
  - `conda install nltk pandas matplotlib`

## Configuration
Variables for configuring both file I/O (namely input and output file locations) and CRAFT network parameters are all consolidated under model/config.py. Please consult the comments in that file for descriptions of the available settings and what they do. Note that the code can be run "as-is" without editing any of the settings, if you want to use the same parameters and data from the CRAFT paper. So, you only need to edit settings.py if (a) you want to customize file paths, or (b) you want to experiment with different network parameters.

## Getting the demo data
If you want to run the code "out-of-the-box" as a demo, you will first need to obtain the training data used in the paper, which is not included in this repository since it exceeds GitHub's file size limits. We have included a script to download the data, simply run:
```
python download_training_data.py
```
Of course, if you have prepared your own custom data as described in the section "Using custom data", you can skip this step (unless you also want the original training data for comparison)

## Running the code

### Pre-training
The first step in getting a functioning CRAFT model up and running is the pre-training phase. As described in the paper, this consists of using large amounts of unlabeled data to train the encoder layers of CRAFT using a generative training objective. Due to the large amounts of data involved, this is the step that will take the longest amount of time; under the default settings, you should expect this to take at least 24 hours and possibly more depending on the power of your GPU.

Pre-training can be run by executing the following command:
```
python train_generative_model.py
```
No command line arguments are needed or taken, as all configuration is read from config.py. As previously stated, you do not need to edit config.py if you simply want to run the standard demo, as all configuration settings have been set to correctly read the input files bundled in this repository and output to a new subdirectory at the top level of the repo directory.

### Fine-tuning and inference
Once the encoder layers have been fitted via pre-training, they can be used as the basis of a classifier model that does the actual forecasting. The process of training such a classifier using the pre-trained starting point is known as fine-tuning. Since fine-tuning uses much smaller amounts of labeled data, it runs much more quickly, so it is presented as an interactive notebook: [`fine_tuning_demo.ipynb`](fine_tuning_demo.ipynb)

## Using custom data

If you want to train a CRAFT model from scratch on your own custom data, rather than simply running the demo of the paper's data, here are the general steps you will need to take.

### Prep your data

CRAFT requires two sets of conversational data: a large amount of unlabeled conversations for pre-training, and a (probably smaller) amount of labeled conversations for fine-tuning. 
While there is no strict requirement for what constitutes a "large" unlabeled dataset, in practice we have found that around 500k or more conversations is a good target for getting decent results.

This implementation uses ConvoKit as its backend for conversational data representation and handling. 
You should ensure that your data is formatted as ConvoKit corpora.
There are no special criteria for the unlabeled dataset, as pre-training only needs to see the text of the utterances, so any corpus should do.
On the other hand, the labeled dataset corpus will need to have the following conversational metadata:
  - A metadata field 'split' which marks the conversation as belonging to the train, test, or validation set. This field should be a string with three possible values: 'train', 'test', or 'val'
  - A metadata field to use as the label. This field can be named anything, but make sure to update `settings.py` accordingly (there is a setting for the name of the label metadata field). This field must be a boolean, where `True` means that the to-be-forecasted event happens in this conversation, and `False` means it does not.

### Update settings.py

Choose a name for your corpus and update the `corpus_name` setting in `settings.py` (this will be used to determine file output locations).
Also remember to update the `label_metadata` setting with the name of the metadata field to use as the label during fine-tuning, as stated in the previous section.

### Process the unlabeled corpus

As previously stated, the unlabeled dataset tends to be large.
Large ConvoKit corpora can tend to be memory-intensive and slow to load, so for efficiency reasons, we first convert the unlabeled corpus to an intermediate JSON lines format that can be more efficiently loaded by the pre-training script.
See [`pretraining_data_prep_demo.ipynb`](pretraining_data_prep_demo.ipynb) for an interactive example of how to do this.

### Extract vocabulary

As a neural model, CRAFT represents words as integers, where each unique vocabulary token maps to a unique integer index.
To construct this mapping from your unlabeled dataset, run the following:
```
python build_vocabulary_objects.py
```
No arguments are needed as all settings are read from `settings.py`

### Run the demo!

Once you have completed all the above steps, you should be all ready to go and can now run CRAFT following the standard demo instructions.

## Questions

Questions and comments may be directed to Jonathan P. Chang (jpc362@cornell.edu), or open an issue on the GitHub page.
