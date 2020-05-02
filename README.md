# python-project-dialog-system-MIAR

## Utrecht University
## Nov 2019

- ###   Riccardo Bassani ([@BassaniRiccardo](https://github.com/BassaniRiccardo))
- ###   Samuel Meyer 
- ###   Geanne Barkmeijer 
- ###   Wiebe de Vries  


This project contains the work done in the course "Methods in AI Research", at Utrecht University.
The project consists of a pyhton implementation of a text-based dialog system, which gives a restaurant recommendation based on the preferences given by the user.
The system classifies user utterances with a decision tree, assigning a dialog act to each utterence. The dialog policy is managed by a finite state function, and the necessary information are retrieved from a database. Further details are available in the report.


## Material:

MAIN PROGRAM: dialog_system.py

 - A working dialog system interface implementing a state transition function (dialog_system.py, line 487)
 - As component of the dialog system: an algorithm identifying user preference statements (dialog_system.py, line 697)
 - As component of the dialog system: a lookup function that retrieves suitable restaurant suggestions from the CSV database (dialog_system.py, line 859) Part 1c
 - An implementation of dialog model configurability (dialog_system, line 13)
 
Further details on the dialog system can be found in Dialog_System_Report.pdf
 
EXPERIMENT PROGRAM: dialog_system_experiment.py
 
 - This file contains a version of the dialog system used to run an experiment on the effects of text-to speech on user satisfaction. Further details on the experiment can be found in Experiment_Report.pdf

ADDITIONAL CODE
 - A Python program that reads the source data and displays the dialogs one by one in human-readable format, using the Enter key to proceed to the next dialog (dialog_printer.py)
 - Python code to produce a text file with one utterance per line in the format dialog_act utterance_content (dialog_system.py, line 389) Part 1b - Python code that implements a keyword matching baseline and a dialog act distribution baseline (dialog_system.py, line 187, 235)


## Configuration info:

The main program runs by default in "user mode" (i.e. showing only the dialog). By setting user mode to False it is possible to perform other actions:
 - get performance of the different classifiers
 - test the different classifiers
 - test difficult instances

This parameter and the dialog configurations can be modified by changing the value of the relative boolean global variables at the top of the code.

Text-to-speech is disabled by default and can be enabled like the other features if it is supported on the device where the program is executed (gtts was used to implement the feature).


