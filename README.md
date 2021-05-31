# Motor Imagery Experiments

People living with disability often require help when using digital technology, an ever increasing necessity in the time of COVID-19. We present BrainBind, a system that allows the disabled more digital independence. At its core BrainBind is a highly flexible Brain Computer Interface (BCI) which can be used by patients with motor impairment to operate a variety of common computer programs. We aim to detect brain activity triggered by imagined movement (the so called Motor Imagery paradigm) using Electroencephalography (EEG). It is of the utmost importance that the system behaves in a user-centric and controllable manner. Therefore we have placed our main focus on designing and testing a robust and accurate classifier which will be used to distinguish the activity resulting from the imagined movement of the left and right hands. Several feature extraction methods and classifiers are tested in different combinations. Data was recorded using an 8 channel (dry electrode) EEG device. The data recorded was prohibitively noisy at times and we were therefore only able to achieve classification accuracies around 60-70 \%. The accuracy decreased further when trying to differentiate between three classes.

In this repository all pipeline steps are implemented. Visualizations were added for data quality reasons and result presentation.

## Research Question

Compare the classification accuracy between:
1. A multiclass classifier
    - Classes: ['idle', 'left', 'right']
2. A multistage binary classifiers
    - Stage 1 classes: ['idle', 'not idle']
    - Stage 2 classes: ['right', 'left']

## Different offline pipelines

![](https://github.com/Jake-Jay/motor-imagery/blob/master/figures/standard-preprocessing.png?raw=true)

During the hackathon and the subsequent weeks we tried several combinations of preprocessing steps, feature extraction methods and classifiers. This repository evaluates their accuracy and false positive rates in several Jupyter Notebooks.
All pipeline experiments were merged into the master with two exceptions. The code for the Two-stage three-class classifier and the binary Idle-MI classifiers can be found in separate branches.

## Results

Our results can be found in the results.md file. The figure below illustrates an lda classifier decision boundary.

![](https://github.com/Jake-Jay/motor-imagery/blob/master/figures/cross-validation-plot.png?raw=true)


## Data
Our own data recordings are in the 'data' folder. For all recordings the Graz recording paradigm was used. Data was recorded using an eight electrode Unicorn BCI device. The different classes are explained during the preprocessing sections in the various Jupyter Notebooks.

Further freely available datasets can be found [here](http://bnci-horizon-2020.eu/database/data-sets).

## Contributors
- Sanda Heshan Lin (TUM)
- Bertram Fuchs (TUM)
- Jake Pencharz (TUM)
