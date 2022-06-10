r"""
This files defines the commands for predicting gesture class based on the EMG data.
"""
import click
from train import train
from inference import inference
from utils import (
    load_emg_recording_dir, 
    load_emg_recording, 
    extract_features, 
    deploy_model,
    load_model
)
from collections import deque

def main() -> None:
    """
    main method of cli.
    """
    run()


@click.group()
def run() -> None:
    """
    group definition for click commands.
    """
    pass


@run.command('train-emg')
@click.option(
    '-t',
    '--train-dir',
    type=click.Path(exists=True, resolve_path=True),
    help='directory that contains the EMG recordings.',
)
@click.option(
    '-d',
    '--deploy-dir',
    type=str,
    default='deployed_models',
    show_default=True,
    help='directory to deploy the train models.',
)
@click.option(
    '-w',
    '--window-size',
    type=int,
    default=100,
    show_default=True,
    help='window size for getting the features from time series.',
)
@click.option(
    '-o',
    '--output-name',
    type=str,
    default=None,
    show_default=True,
    help='name of deployed model file. If not specified, \
        it will be auto generated',
)
def train_emg(
    train_dir: str, deploy_dir: str, window_size: int, output_name: str):
    """ trains a model on data extracted from text files within deploy_dir.
        The results are stored in deploy_dir.
    Args:
        train_dir (str): path to EMG data in txt format. The file must have the
            following columns (`channel1`,`channel2`,`channel3`,`channel4`,`channel5`,
            `channel6`,`channel7`, `channel8`, `class`)
        deploy_dir (str): the directory in which the deployment model should be store.
        window_size (int): to extract features from timeslots in channel signals, 
            some statistics are calculated using a period of time before and after 
            the target timeslot. The length of this period is called window size. 
            This effectively adds a constant window size delay to the inference, 
            which would be negligible if the window size was set to a small value.
        output_name (str): name of deployed model file. If not specified, 
            it will be auto generated'
    """
    # read the data into a data frame
    df = load_emg_recording_dir(train_dir, fillna=0)
    raw_inputs = df.loc[:, ['channel{}'.format(i) for i in range(1, 9)]].values
    raw_labels = df.loc[:, 'class'].values
    # extract features
    features = extract_features(raw_inputs, window_size=window_size)
    labels = raw_labels

    model = train(features, labels)
    deploy_model(model, window_size, deploy_dir, output_name)


@run.command('infer-emg')
@click.option(
    '-r',
    '--recording-path',
    type=click.Path(exists=True, resolve_path=True),
    help='text file that contains the EMG recording (to be replaced by a \
        streaming option in next versions).',
)
@click.option(
    '-m',
    '--model-path',
    type=str,
    help='path to a saved model.',
)
@click.option('-o',
              '--output-path',
              type=str,
              help='path to save the inference results.')
def infer_emg(recording_path: str, model_path: str, output_path: str):
    """predict gesture class based on the EMG data

    Args:
        recording_path (str): path to the EMG data 
        model_path (str): path to a deployment model to load
        output_path (str): path to store result
    """

    
    # load the model
    model, window_size = load_model(model_path)
    out_file = open(output_path, "w")
    
    
    data_buffer = deque(maxlen=window_size * 2)
    # infer the model and print the result
    # TODO: replace part below with an streaming connection (between stars)
    # ****************************************************************
    
    df = load_emg_recording(recording_path, fillna=0)

    # emulate reading from stream
    for i, row in df.loc[
        :, ['channel{}'.format(i) for i in range(1, 9)]].iterrows():
        data_buffer.append(row.values)
        if len(data_buffer) > window_size:
            res = inference(data_buffer, model, window_size)
            if res != 0:
                print('{}->{}'.format(i, res), end=', ', flush=True)
            out_file.write(str(res)+'\n')
    while len(data_buffer) > window_size:
        data_buffer.popleft()
        res = inference(data_buffer, model, window_size)
        if res != 0:
            print('{}->{}'.format(i, res), end=', ', flush=True)
        out_file.write(str(res)+'\n')

    # ****************************************************************
    out_file.close()


if __name__ == '__main__':
    main()
