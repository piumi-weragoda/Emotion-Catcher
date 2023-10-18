import streamlit as st
import os
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import streamlit as st
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import streamlit as st
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os
from json_tricks import dump, load

from pydub import AudioSegment, effects
import librosa
import noisereduce as nr

import tensorflow as tf
import keras
import sklearn
import soundfile as sf
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


def play_audio(audio_file):
    """Play an audio file."""
    wf = wave.open(audio_file, 'rb')
    CHUNK = 1024
    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

    data = wf.readframes(CHUNK)
    while data:
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    audio.terminate()


def record_audio(filename, duration):
    """Record audio input from the user and save it to a file."""
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    audio = pyaudio.PyAudio()

    # start recording
    frames = []
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    # stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save recording to file
    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    st.success(f"Recording saved as {filename}")
    
    
def predict():
    # Importing the model
    saved_model_path = 'C:/Users/janit/OneDrive/Desktop/Janith/modelLSTM.json'
    saved_weights_path = 'C:/Users/janit/OneDrive/Desktop/Janith/modelLSTM_weights.h5'
    # Reading the model from JSON file
    with open(saved_model_path, 'r') as json_file:
        json_savedModel = json_file.read()
        
    model = tf.keras.models.model_from_json(json_savedModel)
    model.load_weights(saved_weights_path)

    model.compile(loss='categorical_crossentropy', 
                    optimizer='RMSProp', 
                    metrics=['categorical_accuracy'])

    emotions = {
        0 : 'Neutral',
        1 : 'Calm',
        2 : 'Happy',
        3 : 'Sad',
        4 : 'Angry',
        5 : 'Fearful',
        6 : 'Disgust',
        7 : 'Suprised'   
    }
    emo_list = list(emotions.values())

    def is_silent(data):
        # Returns 'True' if below the 'silent' threshold
        return max(data) < 100

    #file_path = 'C:/Users/janit/OneDrive/Desktop/Janith/YAF_germ_angry.wav'
    #file_path = 'C:/Users/janit/OneDrive/Desktop/Janith/03-01-03-01-02-02-21.wav'
    file_path = 'C:/Users/janit/OneDrive/Desktop/Janith/user_input.wav'
    array, sr = librosa.core.load(file_path)

    def preprocess(file_path, frame_length = 2048, hop_length = 512):
        '''
        A process to an audio .wav file before execcuting a prediction.
        Arguments:
        - file_path - The system path to the audio file.
        - frame_length - Length of the frame over which to compute the speech features. default: 2048
        - hop_length - Number of samples to advance for each frame. default: 512

        Return:
            'X_3D' variable, containing a shape of: (batch, timesteps, feature) for a single file (batch = 1).
        ''' 
        # Fetch sample rate.
        _, sr = librosa.load(path = file_path, sr = None)
        # Load audio file
        rawsound = AudioSegment.from_file(file_path, duration = None) 
        # Normalize to 5 dBFS 
        normalizedsound = effects.normalize(rawsound, headroom = 5.0) 
        # Transform the audio file to np.array of samples
        normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32') 
        # Noise reduction                  
        final_x = nr.reduce_noise(normal_x, sr=sr)
            
            
        f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length, center=True, pad_mode='reflect').T # Energy - Root Mean Square
        f2 = librosa.feature.zero_crossing_rate(final_x, frame_length=frame_length, hop_length=hop_length,center=True).T # ZCR
        f3 = librosa.feature.mfcc(y=final_x, sr=sr, S=None, n_mfcc=13, hop_length = hop_length).T # MFCC   
        X = np.concatenate((f1, f2, f3), axis = 1)
        
        X_3D = np.expand_dims(X, axis=0)
        # print the shape of the preprocessed array
        print(X_3D.shape)
        data = X_3D

        # Reshape data to have ##400 time steps
        new_time_steps = 400
        # create an empty array with the new shape
        data_reshaped = np.zeros((data.shape[0], new_time_steps, data.shape[2]))  
        # copy the original data into the new array
        data_reshaped[:, :data.shape[1], :] = data 
        # reshape the array to the final shape
        data_reshaped = data_reshaped.reshape((data.shape[0], new_time_steps, data.shape[2]))
        # print the shape of the reshaped array  
        print(data_reshaped.shape)
        
        return data_reshaped

    # load the saved/trained weights
    model.load_weights('C:/Users/janit/OneDrive/Desktop/Janith/modelLSTM_weights.h5')

    # extract features and reshape it
    # features = preprocess(file_path).reshape(1, -1)
    features = preprocess(file_path)
    #print(features.shape)
    predictions = model.predict(features, use_multiprocessing=True)

    total_predictions = []

    # Model's prediction => an 8 emotion probabilities array.
    predictions = model.predict(features, use_multiprocessing=True)
    pred_list = list(predictions)
    print(pred_list)

    # Get rid of 'array' & 'dtype' statments.
    pred_np = np.squeeze(np.array(pred_list).tolist()) 
    print(pred_np)
        
    # Present emotion distribution for a sequence (7.1 secs).
    fig = plt.figure(figsize = (10, 2))
    plt.bar(emo_list, pred_np, color = 'darkturquoise')
    plt.ylabel("Probabilty (%)")
    #plt.show()
    # print the emotion with the maximum probability
    max_emo = np.argmax(predictions)
    print('max emotion:', emotions.get(max_emo,-1))
        
    print(100*'-')
    st.success("EMOTION -->  " + emotions.get(max_emo,-1))

def gender():
    def create_model(vector_length=128):
        """5 hidden dense layers from 256 units to 64, not the best model."""
        model = Sequential()
        model.add(Dense(256, input_shape=(vector_length,)))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        # one output neuron with sigmoid activation function, 0 means female, 1 means male
        model.add(Dense(1, activation="sigmoid"))
        # using binary crossentropy as it's male/female classification (binary)
        model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
        # print summary of the model
        model.summary()
        return model
    
    file_path = 'C:/Users/janit/OneDrive/Desktop/Janith/user_input.wav'
    array, sample_rate = librosa.core.load(file_path)

    def extract_feature(file_name, **kwargs):
        mfcc = kwargs.get("mfcc")
        chroma = kwargs.get("chroma")
        mel = kwargs.get("mel")
        contrast = kwargs.get("contrast")
        tonnetz = kwargs.get("tonnetz")
        # array, sample_rate = librosa.core.load(file_name)
        if chroma or contrast:
            stft = np.abs(librosa.stft(array))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=array, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=array, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(array), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
        return result

    parser = argparse.ArgumentParser(description="""Gender recognition script, this will load the model you trained, 
                                        and perform inference on a sample you provide (either using your voice or a file)""")
    parser.add_argument("-f", "--file", help="The path to the file, preferred to be in WAV format")
    args = parser.parse_args()
    file = args.file
    # construct the model
    model = create_model()
    # load the saved/trained weights
    model.load_weights('C:/Users/janit/OneDrive/Desktop/Janith/model.h5')
    features = extract_feature(file, mel=True).reshape(1, -1)
    # predict the gender!
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    genderP = "Male" if male_prob > female_prob else "Female"
    # show the result!
    print("Result:", genderP)
    print(f"Probabilities:     Male: {male_prob * 100:.2f}%    Female: {female_prob * 100:.2f}%")

    # Results Demostration
    gender = ['Male', 'Female']
    Probability = [male_prob, female_prob]
    # fig = plt.figure(figsize=(2, 3))
    # plt.bar(gender, Probability, color='darkblue')
    # plt.ylabel("Probabilty (%)")
    # plt.title("Percentage Predictions")
    # fig, ax = plt.subplots(figsize=(1, 2))
    # ax.bar(gender, Probability, color='darkblue')
    # ax.set_ylabel("Probability (%)")
    # ax.set_title("Percentage Predictions")
    # st.pyplot(fig)
    st.success("GENDER -->  "+ genderP)

    print(file_path)
    


def app():
    st.set_page_config(page_title="Recognizer",
                       page_icon=":microphone:",
                       layout="wide")

    st.title("EMOTION & GENDER PREDICTOR")

    audio_file = st.file_uploader("Upload a .wav file", type="wav")

    if audio_file is not None:
        filename = "user_input.wav"
        with open(filename, "wb") as f:
            f.write(audio_file.getbuffer())

        st.success(f"File {filename} saved")

    duration = st.slider("Recording duration (seconds):", min_value=1, max_value=10, value=5)

    # Adjust button layout using CSS styling
    st.markdown(
        """
        <style>
        .stButton>button {
            width: 100%;
            box-sizing: border-box;
            padding: 0.375rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create a row layout with three columns
    col1, col2, col3 = st.columns(3)

    # Add the buttons to each column
    with col1:
        record_button = st.button("Record")

    with col2:
        if st.button("Play Recorded Audio"):
            play_audio("user_input.wav")

    with col3:
        if st.button("Clear Recording"):
            if os.path.exists("user_input.wav"):
                os.remove("user_input.wav")
                st.success("Recording deleted")

    status_message = st.empty()

    if record_button:
        status_message.text("Recording...")
        record_audio("user_input.wav", duration)
        status_message.text("Recording finished.")

    if st.button("PREDICT"):
        gender()
        predict()



if __name__ == '__main__':
   app()
