import sys
import numpy as np
import pandas as pd
import datetime
import time
import re
import os
from pprint import pprint
from glob import glob
import threading
import yaml

# External libraries
from PyQt6.QtWidgets import *
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg
from pylsl import StreamInlet, resolve_stream
import playsound
import mne
from transformers import BertTokenizer
from IMAWidgets import PANASWidget, BMISWidget, VAMSWidget, STADIWidget

# Load the configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
pprint(config)

# Print list of available LSL streams
print("Available streams:")
for i, stream in enumerate(resolve_stream()):
    print(f"{i} Stream Name: {StreamInlet(stream).info().name()} \n \
          Stream Type: {StreamInlet(stream).info().type()} \n \
          Stream Channels: {StreamInlet(stream).info().channel_count()} \n \
          Stream Sampling Rate: {StreamInlet(stream).info().nominal_srate()} \n")


def preprocessSentence(sentence, tokenizer):
    """
    Conditional preprocessing on our text unique to our task.
    Args:
    - sentence (str): The sentence to preprocess.
    - tokenizer: The tokenizer to tokenize the sentence.
    - padding (bool): Whether to pad the tokenized sentence or not.

    Returns:
    Tuple of processed sentence, tokens, and token ids.
    """    
    sentence = sentence.lower()

    # Remove words in parenthesis
    sentence = re.sub(r"\([^)]*\)", "", sentence)
    sentence = re.sub(r"'(t|re|ve|m|ll|d|em|s)", r"\1", sentence)

    # Spacing and filters
    sentence = re.sub(r"([-;;.,!?<=>])", r" \1 ", sentence)
    sentence = re.sub("[^A-Za-z0-9]+", " ", sentence) # remove non alphanumeric chars
    
    sentence = re.sub(" +", " ", sentence)  # remove multiple spaces
    sentence = sentence.strip()

    tokens = tokenizer.tokenize(sentence)

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return sentence, tokens, token_ids


class EEGVisualizer(QMainWindow):
    def __init__(self):
        super(EEGVisualizer, self).__init__()

        # Load the sentences
        self.sentences_dir_name = config["SENTENCES_DIR_NAME"]
        self.wav_dir_name = config["WAV_DIR_NAME"]
        self.directory = config["DIRECTORY_OUT"]
        self.sentences_dir = glob(f'{self.sentences_dir_name}/*')
        self.sentences_list = glob(f'{self.sentences_dir[1]}/*.csv')
        self.sentence_listidx = 0
        self.sel_sentences_dir_name = self.sentences_list[self.sentence_listidx].split('/')[1]
        self.sentences = pd.read_csv(self.sentences_list[self.sentence_listidx], sep='\t')['0'].tolist()

        if not os.path.exists(self.directory):
            print(f"{self.directory} Directory does not exist")
            print(f"Creating {self.directory} Directory")
            os.makedirs(self.directory)
            print(f"{self.directory} Directory created")
        else:
            print(f"{self.directory} Directory exists")
        
        # Set up the GUI window
        self.setWindowTitle('EEG Real-time Data Visualization')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.display_length = config["DISPLAY_LENGTH"] 
        self.sample_rate = config["SAMPLE_RATE"]  
        self.WPM = config["WPM"]  
        self.ch_names = config["CHANNELS_NAMES"]
        self.n_channels = len(self.ch_names)
        
        # Define the size of your buffer
        self.buffer_size = self.sample_rate * self.display_length
        # Initialize the buffer
        self.buffer = np.zeros((self.n_channels, self.buffer_size))
        # create mne info object
        self.info = mne.create_info(ch_names=self.ch_names, sfreq=self.sample_rate, ch_types='eeg')

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Set up the list of EEG channels
        self.electrode_list = QListWidget(self) 
        self.electrode_list.addItems(self.ch_names)
        self.electrode_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.electrode_list.itemSelectionChanged.connect(self.update_electrode_visibility)
        self.electrode_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Add this line
        self.layout.addWidget(self.electrode_list)

        # Set up the layout for EEG channels
        self.eeg_layout = QGridLayout()
        self.layout.addLayout(self.eeg_layout)

        # Create a PlotWidget for each channel
        self.plots = [pg.PlotWidget() for _ in range(self.n_channels)]
        for i, plot in enumerate(self.plots):
            row = i // 2  # This will be 0 for the first two plots, 1 for the next two, and so on.
            col = i % 2  # This will alternate between 0 and 1.
            self.eeg_layout.addWidget(plot, row, col)  # Add to the grid layout
            plot.getAxis('left').setLabel(self.ch_names[i]) 
            plot.setYRange(-0.05, 0.05)  # Adjust as needed
            plot.getAxis('bottom').setTicks([])  # Remove x-axis ticks
            if i < self.n_channels - 1:
                plot.getAxis('bottom').setLabel('')  # Remove x-axis label for all but the last plot
            # Trigger the auto-range button functionality
            plot.getViewBox().enableAutoRange()
            
        for electrode in config["DEFAULT_VISIBLE_ELECTRODES"]:
            idx = self.ch_names.index(electrode)
            item = self.electrode_list.item(idx)
            item.setSelected(True)

        # Set up the label for displaying the sentence
        self.current_sentence_index = 0
        self.sentence_label = QLabel()
        self.sentence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sentence_label.setWordWrap(True)
        self.sentence_label.setTextFormat(Qt.TextFormat.RichText)
        font = self.sentence_label.font()
        font.setPointSize(config["FONT_SIZE"])
        # font.setWeight(20)
        self.sentence_label.setFont(font)
        self.layout.addWidget(self.sentence_label, 4)

        self.is_running = False

        self.word_timer = QTimer(self)
        self.word_timer.timeout.connect(self.highlight_next_word)
        
        self.current_word_index = 0
        self.words = self.sentences[self.current_sentence_index].split()

        # Data storage initialization
        self.session_data = []  # This will store each sentence's data as a dictionary

        self.temp_EEG1 = []
        self.temp_trialState = []
        self.temp_trialDelayTimes = []
        self.temp_goTrialEpochs = []
        self.temp_sentenceDurations = []
        self.temp_wordEEG = []

        self.EEG1 = []
        self.trialState = []
        self.trialDelayTimes = []
        self.goTrialEpochs = []
        self.sentenceDurations = []
        self.word_timings_dict = dict()

        # Add control buttons #1
        self.control_layout = QHBoxLayout()
        self.control_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add control buttons #2
        self.control_layout1 = QHBoxLayout()
        # self.control_layout1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add combobox for selecting training mode
        training_choices = [choice.split('/')[-1] for choice in self.sentences_dir]
        self.combobox = QComboBox()
        self.combobox.addItems(training_choices)
        self.combobox.currentIndexChanged.connect(self.training_mode)
        self.control_layout1.addWidget(self.combobox)
    
        self.start_button = QPushButton('Start Session')
        self.start_button.clicked.connect(self.start_session)
        self.control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop Session')
        self.stop_button.clicked.connect(self.stop_session)
        self.control_layout.addWidget(self.stop_button)

        self.restart_button = QPushButton('Restart Session')
        self.restart_button.clicked.connect(self.restart_session)
        self.control_layout.addWidget(self.restart_button)

        self.inspect_button = QPushButton('Inspect EEG')
        self.inspect_button.setCheckable(True)
        self.inspect_button.toggled.connect(self.toggle_eeg_display)
        self.control_layout.addWidget(self.inspect_button)

        # self.ima_button = QPushButton('Show IMA Widgets')
        # self.ima_button.setCheckable(True)
        # self.ima_button.toggled.connect(self.toggle_ima_widgets)
        # self.control_layout.addWidget(self.ima_button)

        # Add combobox for selecting IMA Widgets
        training_choices = ['PANAS', 'BMIS', 'VAMS', 'STADI']
        self.ima = QComboBox()
        self.ima.addItems(training_choices)
        self.ima.currentIndexChanged.connect(self.show_ima_widgets)
        self.control_layout.addWidget(self.ima)
        self.ima_widget = PANASWidget()

        # Add WPM slider
        self.sliderWPM = QSlider(Qt.Orientation.Horizontal)
        self.sliderWPM.setRange(10, 200)  # 1 to 100 for finer granularity; you can adjust as needed
        self.sliderWPM.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sliderWPM.setTickInterval(30)
        self.sliderWPM.setPageStep(10)
        self.sliderWPM.setValue(self.WPM)
        # self.sliderWPM.sliderMoved.connect(self.set_WPM)
        self.sliderWPM.valueChanged.connect(self.set_WPM)
        self.control_layout.addWidget(self.sliderWPM)

        # Add 'Play Audio' button
        self.play_audio_button = QPushButton('Play Audio')
        self.play_audio_button.setCheckable(True)
        self.play_audio_button.toggled.connect(self.toggle_audio_playback)
        self.control_layout1.addWidget(self.play_audio_button)

        self.auto_play_btn = QPushButton("Auto Play Sentence", self)
        self.auto_play_btn.setCheckable(True)
        self.auto_play_btn.toggled.connect(self.toggle_auto_play)
        self.control_layout1.addWidget(self.auto_play_btn) 

        # Add QSpinBox for selecting sentence list index
        self.sentence_idx_spinbox = QSpinBox()
        self.sentence_idx_spinbox.setRange(0, len(self.sentences_list) - 1)
        self.sentence_idx_spinbox.valueChanged.connect(self.load_sentence_list)
        self.sentence_idx_spinbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Add this line
        self.control_layout1.addWidget(QLabel("n Sentences:"))
        self.control_layout1.addWidget(self.sentence_idx_spinbox)

        # Add y-axis range input
        self.yrange_spinbox = QDoubleSpinBox()
        self.yrange_spinbox.setRange(0.0001, 1)  # Adjust range as needed
        self.yrange_spinbox.setSingleStep(0.0001)
        self.yrange_spinbox.setValue(0.00001)
        self.yrange_spinbox.valueChanged.connect(self.update_yrange)
        self.yrange_spinbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Add this line
        self.control_layout1.addWidget(QLabel("Y-axis range:"))
        self.control_layout1.addWidget(self.yrange_spinbox)

        self.layout.addLayout(self.control_layout)
        self.layout.addLayout(self.control_layout1)

        # Load the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["TOKENIZER_TYPE"])

        # LSL stream setup
        try:
            streams = resolve_stream('type', 'EEG')
            self.inlet = StreamInlet(streams[0])
        except IndexError:
            print("No EEG stream found")
        
        self.data = np.zeros((self.n_channels, self.sample_rate*self.display_length))  # 4 seconds of data

        self.display_sentence(self.sentences[self.current_sentence_index])
        self.update_electrode_visibility()
        self.play_audio = False
        self.auto_play = False

        # Update the plots periodically
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(50)  # Refresh rate in milliseconds

    def toggle_auto_play(self, checked):
        self.auto_play = checked

    def show_ima_widgets(self):

        if self.ima.currentText() == 'PANAS':
            self.ima_widget = PANASWidget()
            self.ima_widget.show()
        elif self.ima.currentText() == 'BMIS':
            self.ima_widget = BMISWidget()
            self.ima_widget.show()
        elif self.ima.currentText() == 'VAMS':
            self.ima_widget = VAMSWidget()
            self.ima_widget.show()
        elif self.ima.currentText() == 'STADI':
            self.ima_widget = STADIWidget()
            self.ima_widget.show()
        else:
            self.ima_widget.close()

    def set_WPM(self):
        self.WPM = self.sliderWPM.value()
        
        status = f"WPM set to: {self.WPM}"
        self.statusBar().showMessage(status)
        
    def training_mode(self):
        self.sentences_list = glob(f'{self.sentences_dir_name}/{self.combobox.currentText()}/*.csv')
        self.sentence_listidx = 0
        self.sentences = pd.read_csv(self.sentences_list[self.sentence_listidx], sep='\t')['0'].tolist()
        self.current_sentence_index = 0  # Reset current sentence index
        self.display_sentence(self.sentences[self.current_sentence_index])
        
        # print(self.sentences_list)

    def start_sentence(self):
        self.sentence_start_time = time.time()
        self.current_word_end_time = self.sentence_start_time

    def update(self):
        self.update_eeg_data()

    def update_electrode_visibility(self):
        selected_items = self.electrode_list.selectedItems()
        selected_electrodes = [item.text() for item in selected_items]
        
        for i, plot in enumerate(self.plots):
            if self.ch_names[i] in selected_electrodes:
                plot.setVisible(True)
            else:
                plot.setVisible(False)

    def update_eeg_data(self):
        # Fetch new data
        samples, _ = self.inlet.pull_chunk()
        if samples:
            samples = np.array(samples).T
            samples = np.delete(samples, 25, axis=0)
            self.temp_EEG1.append(samples)  # Store data in temporary list

            # Mock data collection for other variables (replace with actual collection)
            self.data = np.hstack((self.data, samples))[:, -self.sample_rate*self.display_length:]
            
            for plot, channel_data in zip(self.plots, self.data):
                plot.plot(channel_data, clear=True) 

    def toggle_audio_playback(self, checked):
        # Toggle the audio playback flag.
        self.play_audio = checked
    
    def update_yrange(self, value):
        for plot in self.plots:
            plot.setYRange(-value, value)

    def display_sentence(self, sentence):
        self.sentence_label.setText(sentence)
        status = f"Displaying sentence {(self.current_sentence_index + 1):3d} of {len(self.sentences):3d}"
        self.statusBar().showMessage(status)

        self.words = sentence.split()  # Split the sentence into words
        self.current_word_index = 0  # Reset the word index
        
        # Start the word highlighting timer
        self.highlight_next_word()

    def highlight_next_word(self):
        if self.current_word_index < len(self.words):
            word = self.words[self.current_word_index]
            start_time = time.time()  # Store start time
            
            # Calculate word's display duration
            
            scaling_factor = len(word) / config["AVERAGE_WORD_LENGTH"]
            base_duration = 1 / (self.WPM / 60) * 1000 # 1 / (WPM / SECONDS_IN_MINUTE) * MILLISECONDS_IN_SECOND
            word_duration = base_duration * scaling_factor
            self.word_timer.start(int(word_duration))  # Start timer for next word
            
            # Highlight the word in blue bold
            highlighted_sentence = ' '.join(self.words[:self.current_word_index] +
                                            [f"<span style='color: #40E0D0; '>{word}</span>"] +
                                            self.words[self.current_word_index+1:])  # font-weight: bold;
            
            self.sentence_label.setText(highlighted_sentence)
            
            end_time = start_time + word_duration / 1000  # Calculate end time

            self.word_timings_dict[word] = (self.current_word_index, start_time, end_time)  # Store timing information
            self.current_word_index += 1  # Move to the next word
            
        else:
            self.word_timer.stop()  # Stop the timer once all words are read
            
            # Check if auto_play is enabled
            if self.auto_play:
                QTimer.singleShot(250, self.show_next_sentence)

    def show_next_sentence(self):
        # Convert list of 2D arrays to a single 2D array
        eeg_data_for_sentence = np.concatenate(self.temp_EEG1, axis=1)
        sentenceDuration = (eeg_data_for_sentence.shape[1] / self.sample_rate)
        self.adjust_word_timings_to_sentence_duration(sentenceDuration)
        
        sentencePost, tokens, token_ids = preprocessSentence(self.sentences[self.current_sentence_index], self.tokenizer)

        # Create a dictionary for the current sentence's data
        current_data = {
            'EEG1': eeg_data_for_sentence,
            'stateVectors': [],
            'dominantFreqs': [],
            'patchDurations': [],
            'sentenceDuration': sentenceDuration,
            'sentenceOrginal': self.sentences[self.current_sentence_index],
            'sentencePost': sentencePost,  
            'Tokens': tokens,
            'tokenIds': token_ids,
            'wordTimings': self.word_timings_dict,
            'WPM': self.WPM,
        }

        self.word_timings_dict = dict()

        # Append the current data dictionary to the session data
        self.session_data.append(current_data)

        # pprint(self.session_data)
        
        # Clear the temporary data
        self.temp_EEG1.clear()

        # Move to the next sentence
        self.current_sentence_index += 1
        if self.current_sentence_index >= len(self.sentences):
            self.current_sentence_index = 0
        self.display_sentence(self.sentences[self.current_sentence_index])
        
        # If play_audio flag is set, play the audio corresponding to the current sentence
        if self.play_audio:
            audio_file_path = f"{self.wav_dir_name}/{self.sel_sentences_dir_name}/{self.sentences[self.current_sentence_index].replace(' ','_')[:-1]}.wav"
            threading.Thread(target=playsound.playsound, args=(audio_file_path,)).start()

    def adjust_word_timings_to_sentence_duration(self, sentenceDuration):
        """Adjust the word timings to fit the sentence duration."""

        # Sort the dictionary based on word_id
        sorted_word_timings = sorted(self.word_timings_dict.items(), key=lambda x: x[1][0])

        # Find the start time of the first word and the end time of the last word
        original_sentence_start_time = sorted_word_timings[0][1][1]
        original_sentence_end_time = sorted_word_timings[-1][1][2]

        # Calculate the original sentence duration
        original_sentence_duration = original_sentence_end_time - original_sentence_start_time

        # Calculate the scaling factor
        scaling_factor = sentenceDuration / original_sentence_duration

        adjusted_word_timings = {}
        for word, (word_id, start_time, end_time) in sorted_word_timings:
            # Adjust timings so that the first word starts at 0
            adjusted_start_time = start_time - original_sentence_start_time
            adjusted_end_time = end_time - original_sentence_start_time
            
            # Scale the adjusted timings
            adjusted_start_time *= scaling_factor
            adjusted_end_time *= scaling_factor

            # Store the adjusted timings
            adjusted_word_timings[word] = (word_id, adjusted_start_time, adjusted_end_time)

        adjusted_word_timings = sorted(adjusted_word_timings.items(), key=lambda x: x[1][0])  # Sort the dictionary based on word_id
        self.word_timings_dict = adjusted_word_timings

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.show_next_sentence()

    def start_session(self):
        if not self.is_running:
            self.is_running = True
            self.EEG1.clear()
            self.trialState.clear()
            self.trialDelayTimes.clear()
            self.goTrialEpochs.clear()
            self.sentenceDurations.clear()
            self.timer.start()

    def stop_session(self):
        if self.is_running:
            self.is_running = False
            self.timer.stop()
            self.word_timer.stop()
            
            # Collect the results od ima widgets and add them to the session data
            if self.ima.currentText() == 'PANAS' and self.ima_widget is not None:
                self.session_data[-1]['PANAS'] = self.ima_widget.collect_data()
            elif self.ima.currentText() == 'BMIS' and self.ima_widget is not None:
                self.session_data[-1]['BMIS'] = self.ima_widget.collect_data()
            elif self.ima.currentText() == 'VAMS' and self.ima_widget is not None:
                self.session_data[-1]['VAMS'] = self.ima_widget.collect_data()
            elif self.ima.currentText() == 'STADI' and self.ima_widget is not None:
                self.session_data[-1]['STADI'] = self.ima_widget.collect_data()

            # Save the session data list to a numpy file
            now = datetime.datetime.now()
            sentence = self.sentences_list[self.sentence_listidx] 
            filename_to_save = f"{self.directory}/{sentence.split('/')[-1].split('.')[0]}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.npy"
            # print(f"sessionData/{sentence.split('/')[-1].split('.')[0]}_{now.strftime('%Y-%m-%d_%H-%M-%S')}")
            np.save(filename_to_save, self.session_data[1:])
            print(f"Session data saved to {filename_to_save}")
            self.session_data.clear()


    def restart_session(self):
        self.stop_session()
        # self.start_session()
        self.session_data.clear()
        self.current_sentence_index = 0
        self.display_sentence(self.sentences[self.current_sentence_index])
        self.auto_play_btn.setChecked(False)
        self.play_audio_button.setChecked(False)

    def toggle_eeg_display(self, checked):
        # Toggle EEG display
        self.electrode_list.setVisible(checked)
        # self.plots.show()
        for plot in self.plots:
            plot.setVisible(checked)

    def load_sentence_list(self, idx):
        """Load sentences from the selected sentence list index."""
        self.sentence_listidx = idx
        self.sentences_list = sorted(self.sentences_list)
        self.sentences = pd.read_csv(self.sentences_list[self.sentence_listidx], sep='\t')['0'].tolist()
        self.current_sentence_index = 0  # Reset current sentence index
        self.display_sentence(self.sentences[self.current_sentence_index])

        status = f"Displaying {self.sentences_list[self.sentence_listidx]}"
        self.statusBar().showMessage(status)


app = QApplication(sys.argv)
window = EEGVisualizer()
window.show()
sys.exit(app.exec())
