from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from music21 import converter
import numpy as np
import torch


# Load model, scaler, and label encoder once to reuse for multiple requests
MODEL_PATH = "best.pth"
SCALAR_MEAN_PATH = "scalar_mean.npy"
SCALAR_SCALE_PATH = "scalar_scale.npy"
LABEL_CLASSES_PATH = "label_classes.npy"


class ComposerNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ComposerNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output_size)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def extract_tempo(abc_notation):
    return 10

def extract_duration(abc_notation):
    return 300

def extract_key_signature(abc_notation):
    return 200

def processing_pipeline(tunes_tuples):
    """
    Processes a list of tunes, extracting musical features from each ABC notation.

    Parameters:
    -----------
    tunes_tuples : list of tuples
        Each tuple contains (tune_name, tune_composer, abc_notation).

    Returns:
    --------
    dict
        A dictionary where each key is a tune name, and each value is a dictionary of extracted features, 
        including the tune's composer.
    """
    tunes_extracted_features = {}
    for tune_name, tune_composer, abc_notation in tunes_tuples:
        try:
            midi = converter.parse(abc_notation.strip())
            features = extract_features(midi)
            tunes_extracted_features[tune_name] = features
            tunes_extracted_features[tune_name]['composer'] = tune_composer
        except:
            print(f"Warning: An error occurred for tune: {tune_name}; composer: {tune_composer}")
        
    return tunes_extracted_features

def extract_features(midi_format):
    """
    Extracts musical features from a MIDI format object.

    Parameters:
    -----------
    midi_format : music21.stream.Score
        The parsed MIDI object from an ABC notation.

    Returns:
    --------
    dict
        A dictionary of extracted features, including counts of notes, rests, chords, 
        and statistical data on pitch, duration, and intervals.
    """
    pitches = []
    durations = []
    rests = 0
    chords = 0
    intervals = []
    notes = 0

    for element in midi_format.flat.notes:
        notes += 1
        if element.isRest:
            rests += 1
        elif element.isChord:
            chords += 1
        else:
            pitches.append(element.pitch.midi)
            durations.append(element.duration.quarterLength)
            intervals.append(element.pitch.ps)

    avg_pitch = sum(pitches) / len(pitches) if pitches else 0
    pitch_range = max(pitches) - min(pitches) if pitches else 0
    pitch_sd = np.std(pitches) if pitches else 0
    pitches_len = len(pitches) if pitches else 0
    
    avg_duration = sum(durations) / len(durations) if durations else 0
    duration_range = max(durations) - min(durations) if durations else 0
    duration_sd = np.std(durations) if durations else 0
    total_duration = sum(durations) if durations else 0

    avg_interval = sum(intervals) / len(intervals) if intervals else 0
    interval_range = max(intervals) - min(intervals) if intervals else 0
    interval_sd = np.std(intervals) if intervals else 0

    features = {
        'notes': notes,
        'rests': rests,
        'chords': chords,
        'avg_pitch': avg_pitch,
        'pitch_range': pitch_range,
        'pitch_sd': pitch_sd,
        'pitches_len': pitches_len,
        'avg_duration': avg_duration,
        'duration_range': duration_range,
        'duration_sd': duration_sd,
        'total_duration': total_duration,
        'avg_interval': avg_interval,
        'interval_range': interval_range,
        'interval_sd': interval_sd
    }

    return features


def convert_abc_to_midi(abc_tunes):
    midi_tunes = {}
    for composer, abc_tunes in abc_tunes.items():
        for abc_tune in abc_tunes:
            midi = converter.parse(abc_tune)
            if composer not in midi_tunes:
                midi_tunes[composer] = [midi]
            else:
                midi_tunes[composer].append(midi)
    return midi_tunes


def preprocess_abc_for_nn(abc_notation):
    """
    Preprocess ABC notation into standardized feature vector.

    Args:
        abc_notation (str): ABC notation string.

    Returns:
        np.ndarray: Standardized feature vector.
    """
    # Convert ABC notation to MIDI
    midi_tune = convert_abc_to_midi(abc_notation)
    composer_num = len(midi_tune.keys())
    print("There are", composer_num, "composers.")
    
    # Extract features
    features = extract_features([midi_tune])['unknown']

    # Create a feature vector
    feature_vector = np.array([
        features['avg_pitch'],
        features['pitch_range'],
        features['pitch_sd'],
        features['avg_duration'],
        features['duration_range'],
        features['duration_sd'],
        features['avg_interval'],
        features['interval_range'],
        features['interval_sd']
    ]).reshape(1, -1)  # Reshape to 2D array for scaling

    # Standardize the feature vector
    standardized_vector = scalar.transform(feature_vector)
    return standardized_vector, composer_num


def get_inference(abc_notation):
    """
    Perform inference to classify the composer from ABC notation.

    Args:
        abc_notation (str): ABC notation string.

    Returns:
        str: Predicted composer name.
    """
    feature_vector, composer_num = preprocess_abc_for_nn(abc_notation)

    # Load the model
    input_size = 9  # Number of features used during training
    model = ComposerNN(input_size, output_size=composer_num)  # Adjust output_size if needed
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load the StandardScaler and LabelEncoder
    scalar = StandardScaler()
    scalar.mean_ = np.load(SCALAR_MEAN_PATH)
    scalar.scale_ = np.load(SCALAR_SCALE_PATH)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(LABEL_CLASSES_PATH)

    
    with torch.no_grad():
        input_tensor = torch.tensor(feature_vector, dtype=torch.float32)
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
        return label_encoder.inverse_transform([predicted_label])[0]