from PyQt6.QtWidgets import (QVBoxLayout, QLabel, QFormLayout, QWidget,
                             QSlider, QPushButton, QHBoxLayout, QComboBox)
from PyQt6.QtCore import Qt

class PANASWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        # Mood labels for PANAS
        self.moods = {
            "interested": None,
            "distressed": None,
            "excited": None,
            "upset": None,
            "strong": None,
            "guilty": None,
            "scared": None,
            "hostile": None,
            "enthusiastic": None,
            "proud": None,
            "irritable": None,
            "alert": None,
            "ashamed": None,
            "inspired": None,
            "nervous": None,
            "determined": None,
            "attentive": None,
            "jittery": None,
            "active": None,
            "afraid": None
        }

        self.form_layout = QFormLayout()

        for mood, _ in self.moods.items():
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(1, 5)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(1)
            self.moods[mood] = slider
            self.form_layout.addRow(QLabel(mood.capitalize()), slider)

        self.layout.addLayout(self.form_layout)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.collect_data)
        self.layout.addWidget(self.submit_button)

        self.setLayout(self.layout)

    def collect_data(self):
        data = {}
        for mood, slider in self.moods.items():
            data[mood] = slider.value()
        return data  # Return the data for further use


class BMISWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        # Mood adjectives for BMIS
        self.mood_adjectives = {
            "Happy": None,
            "Sad": None,
            "Calm": None,
            "Nervous": None,
            "Lively": None,
            "Bored": None,
            "Restless": None,
            "Peaceful": None,
            "Drowsy": None,
            "Attentive": None,
            "Jittery": None,
            "Introspective": None,
            "Active": None,
            "Ruminating": None,
            "Cheerful": None,
            "Upset": None
        }

        self.form_layout = QFormLayout()

        rating_choices = ["", "Definitely do not feel", "Do not feel", "Slightly feel", "Definitely feel"]

        for mood, _ in self.mood_adjectives.items():
            combobox = QComboBox()
            combobox.addItems(rating_choices)
            self.mood_adjectives[mood] = combobox
            self.form_layout.addRow(QLabel(mood), combobox)

        self.layout.addLayout(self.form_layout)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.collect_data)
        self.layout.addWidget(self.submit_button)

        self.setLayout(self.layout)

    def collect_data(self):
        data = {}
        for mood, combobox in self.mood_adjectives.items():
            data[mood] = combobox.currentText()
        return data  # Return the data for further use

class VAMSWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        # Mood descriptors for VAMS (you can customize this list as needed)
        self.mood_descriptors = {
            "happy-sad": None,
            "relaxed-anxious": None,
            "energetic-tired": None
            # ... add any other descriptor pairs here
        }

        self.form_layout = QFormLayout()

        for descriptor, _ in self.mood_descriptors.items():
            extremes = descriptor.split('-')
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(1, 100)  # 1 to 100 for finer granularity; you can adjust as needed
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(10)  # every 10 units

            hlayout = QHBoxLayout()
            hlayout.addWidget(QLabel(extremes[0]))
            hlayout.addWidget(slider)
            hlayout.addWidget(QLabel(extremes[1]))

            self.mood_descriptors[descriptor] = slider
            self.form_layout.addRow(QLabel("Rate your mood:"), hlayout)

        self.layout.addLayout(self.form_layout)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.collect_data)
        self.layout.addWidget(self.submit_button)

        self.setLayout(self.layout)

    def collect_data(self):
        data = {}
        for descriptor, slider in self.mood_descriptors.items():
            data[descriptor] = slider.value()
        return data  # Return the data for further use

class STADIWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        # Hypothetical questions for STADI (please replace with the actual questions)
        self.questions = {
            "I often feel nervous": None,
            "I regularly feel sad": None,
            # ... add all other questions here
        }

        self.form_layout = QFormLayout()

        for question, _ in self.questions.items():
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(1, 5)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(1)
            self.questions[question] = slider
            self.form_layout.addRow(QLabel(question), slider)

        self.layout.addLayout(self.form_layout)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.collect_data)
        self.layout.addWidget(self.submit_button)

        self.setLayout(self.layout)

    def collect_data(self):
        data = {}
        for question, slider in self.questions.items():
            data[question] = slider.value()
        return data  # Return the data for further use
