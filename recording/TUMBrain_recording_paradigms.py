# -*- coding: utf-8 -*-
from psychopy import visual, event, sound
from psychopy.core import StaticPeriod
from pylsl import StreamInfo, StreamOutlet
import os, random


# ToDo: implement exit handler, for now dont use fullscr mode unless you are sure you want to do the entire experiment

class Paradigm:
    """
    Class for running the paradigm.
    """

    def __init__(self, window):
        self.window = window
        self.fixation = visual.ShapeStim(win=window,
                                         vertices=((0, -0.05), (0, 0.05), (0, 0), (-0.05, 0), (0.05, 0)),
                                         lineWidth=2,
                                         closeShape=False,
                                         lineColor="white")

    def run_block(self, cue, markerstream):
        """
        For a single motor imagery, it shows the cue, fixation, and break.
        The duration of each subperiod is defined in the dictionary object

        Parameters
        ----------

        cue: Cue object (either ImageCue, WordCue or SubstractionCue)
        markerstream: markerstream sends markers to pylsl
        """
        # create image
        image = cue.get_image(self.window)

        # cue
        isi = StaticPeriod(screenHz=60)
        isi.start(duration['cue'])
        image.draw()
        self.window.flip()
        markerstream.push_sample(['cue_' + cue.name])
        isi.complete()

        # imagery
        isi = StaticPeriod(screenHz=60)
        isi.start(duration['motor'])
        self.fixation.draw()
        self.window.flip()
        markerstream.push_sample(['imagery_' + cue.name])
        print(['imagery_' + cue.name])
        isi.complete()

        # break
        isi = StaticPeriod(screenHz=60)
        isi.start(duration['break'])
        self.window.flip()
        markerstream.push_sample(['break'])
        isi.complete()

    def run_set(self, cues, num_trials, repetition=1):

        """
        Run one set of cues num_trials*repetition times.

        Parameters
        ----------

        cues: list of cue objects that are supposed to be shown
        num_trials: number of trials (there is a break between trials)
        repetition: number of reptition without break
        """

        # define stream
        info = StreamInfo(name='openvibeMarkers', type='Markers', channel_count=1,
                          nominal_srate=0,
                          channel_format='string',
                          source_id='cybathlon_MI_markers')

        markerstream = StreamOutlet(info)

        message_start = visual.TextStim(win=self.window, text='Experiment starts after Beep, press any key to continue',
                                        pos=(0, 0), units='norm')
        message_start.draw()
        self.window.flip()
        event.waitKeys()

        # load sound
        beep = sound.Sound(os.path.join('figures', 'beep.wav'))

        random.shuffle(cues)
        isi = StaticPeriod(screenHz=60)
        isi.start(duration['beep'])
        beep.play()
        markerstream.push_sample(['beep'])
        isi.complete()
        self.run_block(ImageCue('no_blink'), markerstream)

        stay = True
        while stay:
            # starts trial
            for n in range(num_trials):
                # shuffle order

                if n > 0:
                    isi = StaticPeriod(screenHz=60)
                    isi.start(duration['beep'])
                    beep.play()
                    markerstream.push_sample(['beep'])
                    isi.complete()

                for _ in range(repetition):
                    random.shuffle(cues)

                    for motor in cues:
                        self.run_block(motor, markerstream)

                message_break = visual.TextStim(win=self.window, text='Press any Key to Continue with the next Trial',
                                                pos=(0, 0),
                                                units='norm')
                message_break.draw()
                self.window.flip()
                event.waitKeys()

            while True:
                message_new_trial = visual.TextStim(win=self.window,
                                                    text='Press Key UP to Start New Trial, or DOWN to close.',
                                                    pos=(0, 0), units='norm')
                message_new_trial.draw()
                self.window.flip()

                keys = event.getKeys(['up', 'down'])
                if 'down' in keys:
                    stay = False
                    break
                elif 'up' in keys:
                    break

    # four paradigms in a group

    def graz_experiment(self):
        """Run Graz Paradigm"""

        self.run_set([ImageCue('auditory'), ImageCue('foot'), SubstractionCue(), ImageCue('pause')], num_trials=2)
        self.run_set([ImageCue('handL'), ImageCue('foot'), SubstractionCue(), ImageCue('pause')], num_trials=2)
        self.run_set([ImageCue('handL'), ImageCue('spatial'), WordCue(), ImageCue('pause')], num_trials=2)
        self.run_set([ImageCue('handL'), ImageCue('auditory'), ImageCue('taste'), ImageCue('pause')], num_trials=2)

        message_end = visual.TextStim(win=self.window, text='Thanks for participation!', pos=(0, 0), units='norm')
        message_end.draw()
        self.window.flip()
        event.waitKeys()

    def two_paradigm_experiment(self, repetition=5, fullscr=False):
        """
        Run two cue paradigm.

        Paramters
        ---------
        repetition: repeat each paradigm in run_set()
        fullscr: False if debugging, True if Recording
        """
        self.window = visual.Window(units='height', fullscr=fullscr)
        self.run_set([ImageCue('hand'), ImageCue('foot')], num_trials=10, repetition=repetition)
        message_end = visual.TextStim(win=self.window, text='Thanks for participation!', pos=(0, 0), units='norm')
        message_end.draw()
        self.window.flip()
        event.waitKeys()

    def three_paradimg_experiment(self, repetition=1, fullscr=False):
        """
        Run experiment with three different cues.

        Parameters
        ----------
        repetition: repeat each paradigm in run_set()
        fullscr: False if debugging, True if Recording
        """
        self.window = visual.Window(units='height', fullscr=fullscr)
        self.run_set([ImageCue('handL'), ImageCue('handR'), ImageCue('foot')], num_trials=5, repetition=repetition)
        message_end = visual.TextStim(win=self.window, text='Thanks for participation!', pos=(0, 0), units='norm')
        message_end.draw()
        self.window.flip()
        event.waitKeys()


# different types of cues are implemented below (more could be added, they need a name and a get_image function):
class ImageCue:
    """
    Class to create cues that are based on images.
    """

    def __init__(self, name):
        self.name = name
        self.stimuli_path = os.path.join('figures', name + '.png')
        # todo: would be nice to have check if path is extant

    def get_image(self, win):
        image = visual.ImageStim(win, self.stimuli_path)
        return image


class SubstractionCue:
    """
    Class to create substraction exercises as cues.
    """

    def __init__(self):
        self.name = "subs"

    def get_image(self, win):
        num1 = str(random.randint(50, 99))
        num2 = str(random.randint(6, 9))
        image = visual.TextStim(win=win, text=num1 + '-' + num2)
        return image


class WordCue:
    """
    Class to create letter cues.
    """

    def __init__(self):
        self.name = "word"

    def get_image(self, win):
        letter = random.choice(['A', 'E', 'F', 'M', 'D', 'I', 'L'])
        image = visual.TextStim(win=win, text=letter)
        return image


if __name__ == '__main__':
    '''
    Available stimuli:

    ImageCue('handL'),
    ImageCue('handR'),
    ImageCue('foot'),
    ImageCue('breathe'),
    WordCue(),
    SubstractionCue(),
    ImageCue('spatial'),
    ImageCue('auditory'),
    ImageCue('no_blink'),
    ImageCue('pause')
    '''

    duration = {
        'motor': 4,
        'beep': 1,
        'break': 4,
        'cue': 2
    }

    stimuli = [
        ImageCue('handL'),
        ImageCue('handR')
    ]

    mywin = visual.Window(units='height', fullscr=False)
    paradigm: Paradigm = Paradigm(mywin)
    paradigm.run_set(stimuli, num_trials=20, repetition=15)

