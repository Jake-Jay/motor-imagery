import matplotlib.pyplot as plt
import util.preprocess_util as preprocess
import matplotlib.transforms as mtransforms
import numpy as np


def print_results(classes, accuracies:list, confusion_matrices:list, format=None):
    nfolds = len(accuracies)

    if format is None:
        print('{}\tTrue {}\tFalse {}\tFalse {}\tTrue {}\n{}'.format(
        'Accuracy', 
        classes[0], 
        classes[0],
        classes[1], 
        classes[1],
        '-'*110
        ))

        av_acc = 0
        av_conf = np.zeros([2,2])
        for i in range(nfolds):
            print('{:.2f}\t\t\t{}\t\t\t{}\t\t\t{}\t\t\t{}'.format(
                accuracies[i],
                confusion_matrices[i][0,0],
                confusion_matrices[i][0,1],
                confusion_matrices[i][1,0],
                confusion_matrices[i][1,1])
            )
            av_acc += accuracies[i]
            av_conf += confusion_matrices[i]

        av_acc /= nfolds
        av_conf = av_conf/nfolds
        print('{}\n{:.2f}\t\t\t{}\t\t\t{}\t\t\t{}\t\t\t{}\n{}'.format(
            '-'*110,
            av_acc,
            av_conf[0,0],
            av_conf[0,1],
            av_conf[1,0],
            av_conf[1,1],
            '-'*110
        ))
    
    elif format=='tex':
        print('&{}\t& True {}\t& False {}\t& False {}\t& True {} {}'.format(
            'Accuracy', 
            classes[0], 
            classes[0],
            classes[1], 
            classes[1],
            r'\\'
        ))

        av_acc = 0
        av_conf = np.zeros([2,2])
        for i in range(nfolds):
            print('Fold {} & {:.2f}\t\t& {}\t\t& {}\t\t& {}\t\t& {} {}'.format(
                i,
                accuracies[i],
                confusion_matrices[i][0,0],
                confusion_matrices[i][0,1],
                confusion_matrices[i][1,0],
                confusion_matrices[i][1,1],
                r'\\'
            ))
            av_acc += accuracies[i]
            av_conf += confusion_matrices[i]

        av_acc /= nfolds
        av_conf = av_conf/nfolds
        print('Average &{:.2f}\t\t& {}\t\t& {}\t\t& {}\t\t& {} {}'.format(
            av_acc,
            av_conf[0,0],
            av_conf[0,1],
            av_conf[1,0],
            av_conf[1,1],
            r'\\'
        ))
    
    elif format == 'md':
        print('Fold |{}\t| True {}\t| False {}\t| False {}\t| True {} |'.format(
            'Accuracy', 
            classes[0], 
            classes[0],
            classes[1], 
            classes[1]
        ))
        print('--- |'*6)

        av_acc = 0
        av_conf = np.zeros([2,2])
        for i in range(nfolds):
            print('Fold {} | {:.2f}\t\t| {}\t\t| {}\t\t| {}\t\t| {} |'.format(
                i,
                accuracies[i],
                confusion_matrices[i][0,0],
                confusion_matrices[i][0,1],
                confusion_matrices[i][1,0],
                confusion_matrices[i][1,1],
            ))
            av_acc += accuracies[i]
            av_conf += confusion_matrices[i]

        av_acc /= nfolds
        av_conf = av_conf/nfolds
        print('Average |{:.2f}\t\t| {}\t\t| {}\t\t| {}\t\t| {} |'.format(
            av_acc,
            av_conf[0,0],
            av_conf[0,1],
            av_conf[1,0],
            av_conf[1,1],
        ))




def plot_events(
        relative_eeg_timestamps,
        event_time_series,
        fs,
        label2code,
        window=100,
        start_time=15):


    x = relative_eeg_timestamps[start_time*fs:(start_time + window)*fs]
    y = event_time_series[start_time*fs:(start_time + window)*fs]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x, y)

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    ax.fill_between(x, 0, 1, where= (y == 5),
                    facecolor='green', alpha=0.4, transform=trans, label='Right Hand')
    ax.fill_between(x, 0, 1, where= (y == 7),
                    facecolor='orange', alpha=0.5, transform=trans, label='Left Hand')

    event_labels = list(label2code.keys())
    plt.yticks(range(len(event_labels)), event_labels)
    plt.ylim([2.8,7.2])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), shadow=True, fancybox=True)
    plt.tight_layout()
    plt.show()


def plot_all(
        eeg_timestamps, 
        event_time_series, 
        eeg_data, 
        event_labels,
        fs,
        number_of_channels=None, 
        window=30, 
        start_time = 15):

    if number_of_channels == None:
        number_of_channels = eeg_data.shape[0]

    colors = plt.rcParams["axes.prop_cycle"]()

    height_ratios = [1 if i>0 else 3 for i in range(number_of_channels+1)]

    fig, axs = plt.subplots(
        number_of_channels+1, 1,
        figsize=(10,8), 
        sharex=True, 
        gridspec_kw={'height_ratios': height_ratios}
    )
    plt.tight_layout()

    x = eeg_timestamps[start_time*fs:(start_time + window)*fs]
    y = event_time_series[start_time*fs:(start_time + window)*fs]

    c = next(colors)["color"]
    axs[0].plot(x, y, color=c)
    axs[0].set_title('Marker Stream')
    plt.sca(axs[0])
    plt.yticks(range(len(event_labels)), event_labels)
    axs[0].set_ylim([2.8,7.2])

    for i in range(number_of_channels):
        axs[i+1].plot(
            x, 
            eeg_data[i ,start_time*fs:(start_time + window)*fs], 
            color=next(colors)["color"]
        )
        axs[i+1].set_title(preprocess.channel2name(i))
        axs[i+1].set_ylabel(r'$\mu$ V')

    plt.show()