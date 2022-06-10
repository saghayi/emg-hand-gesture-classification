# Introduction

Hand gestures classification based on the dataset shared on [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets) accessible at http://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures. 

# Dataset
The dataset contains Electromyography (EMG) recording from the wrist of 36 subjects who were asked to perform one of the 7 predetermined hand gestures. The recording constitutes of 8 channels each corresponding to a sensor. Every subject is asked to perform all of the first 6 gestures while only one subject is also asked to perform the last gesture. The data is structured and saved in CSV-like format. Entries are separated with tab('\t'). The columns are `time`, `channel1` to `channel8`, `class`. The column `class` represents the gesture category. Class zero corresponds to the resting state (no gesture).   


# Installation
Locate at the root of the project in your terminal where you can see `requirements.txt`, then install the requirements using `pip`

```bash
 $ pip install -r requirements.txt
 ```

### Training
```bash
$ python cli.py train-emg -t <path_to_dir_containing_EMG_recordings> -w <window_size> -d <dir_to_save_trained_model> -o <name_of_output_file>
```
Example:
```bash
$ python cli.py train-emg -t .\data\EMG_data_for_gestures-master\01\ -w 100 -d deployed_models -o train_on_01.pkl
```


### Inference
```bash
$ python cli.py infer-emg -r <path_to_dir_containing_EMG_recordings> -m <path_to_dir_containing_trained_model> -o <path_to_save_the_inference_results>
```
Example:
```bash
$ python cli.py infer-emg -r  data/EMG_data_for_gestures-master/17/2_raw_data_11-20_23.03.16.txt -m deployed_models/RandomForestClassifier_w100___2022-06-02.pkl -o inference_output/inference17_2.txt
```