#!/usr/bin/env python
# coding: utf-8

# In[181]:


# Probably more imports than are really necessary...
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from tqdm import tqdm
import librosa
import numpy as np
import miditoolkit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import random


# ## Metrics

# In[277]:


def accuracy1(groundtruth, predictions):
    correct = 0
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        if predictions[k] == groundtruth[k]:
            correct += 1
    return correct / len(groundtruth)


# In[279]:


def accuracy2(groundtruth, predictions):
    correct = 0
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        if predictions[k] == groundtruth[k]:
            correct += 1
    return correct / len(groundtruth)


# In[281]:


TAGS = ['rock', 'oldies', 'jazz', 'pop', 'dance',  'blues',  'punk', 'chill', 'electronic', 'country']


# In[283]:


def accuracy3(groundtruth, predictions):
    preds, targets = [], []
    for k in groundtruth:
        if not (k in predictions):
            print("Missing " + str(k) + " from predictions")
            return 0
        prediction = [1 if tag in predictions[k] else 0 for tag in TAGS]
        target = [1 if tag in groundtruth[k] else 0 for tag in TAGS]
        preds.append(prediction)
        targets.append(target)
    
    mAP = average_precision_score(targets, preds, average='macro')
    return mAP


# ## Task 1: Composer classification

# In[286]:


dataroot1 = "student_files/task1_composer_classification/"


# In[332]:


class model1():
    def __init__(self):
        pass

    def features(self, path):
        filepath = path if os.path.exists(path) else os.path.join(dataroot1, path)
        midi_obj = miditoolkit.midi.parser.MidiFile(filepath)
        # midi_obj = miditoolkit.midi.parser.MidiFile(dataroot1 + '/' + path)
        notes = midi_obj.instruments[0].notes
        if not notes:
            return [0] * 4 + [0] * 12 # handle empty case

        num_notes = len(notes)
        pitches = [note.pitch for note in notes]
        durations = [note.end - note.start for note in notes]
        total_time = max(note.end for note in notes) - min(note.start for note in notes)
        
        average_pitch = sum([note.pitch for note in notes]) / num_notes
        std_pitch = np.std([note.pitch for note in notes])
        average_duration = sum([note.end - note.start for note in notes]) / num_notes
        std_duration = np.std(durations)
        note_density = num_notes / total_time if total_time > 0 else 0
        pitch_range = max(pitches) - min(pitches)

        # Pitch class histogram
        pitch_classes = [note.pitch % 12 for note in notes]
        hist = [0] * 12
        for pc in pitch_classes:
            hist[pc] += 1
        hist = [x / num_notes for x in hist] # normalize

        # interval histogram
        intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
        hist_intervals = [0] * 5
        for interval in intervals:
            if interval < -6:
                hist_intervals[0] += 1
            elif interval < 0:
                hist_intervals[1] += 1
            elif interval == 0:
                hist_intervals[2] += 1
            elif interval <= 6:
                hist_intervals[3] += 1
            else:
                hist_intervals[4] += 1

        if intervals:
            hist_intervals = [x / len(intervals) for x in hist_intervals]
        else:
            hist_intervals = [0] * 5


        # symbolic features
        unique_pitches = len(set(pitches))
        start_times = [note.start for note in notes]
        unique_starts = len(set(start_times))
        articulation_rate = unique_starts / total_time if total_time > 0 else 0

        sorted_notes = sorted(notes, key=lambda n: n.start)
        rest_time = 0.0
        for i in range(1, len(sorted_notes)):
            prev_end = sorted_notes[i-1].end
            curr_start = sorted_notes[i].start
            if curr_start > prev_end:
                rest_time += curr_start - prev_end
        rest_ratio = rest_time / total_time if total_time > 0 else 0

        # polyphony
        start_time_counts = {}
        for note in notes:
            t = round(note.start, 3)
            start_time_counts[t] = start_time_counts.get(t,0) + 1
        polyphonic_events = sum(1 for count in start_time_counts.values() if count > 1)
        polyphony_ratio = polyphonic_events / len(start_time_counts) if start_time_counts else 0

        # unique duration count
        unique_durations = len(set(round(d, 3) for d in durations))

        # longest rest
        longest_rest = 0.0
        for i in range(1, len(sorted_notes)):
            gap = sorted_notes[i].start - sorted_notes[i-1].end
            if gap > longest_rest:
                longest_rest = gap


        # note start times
        onset_std = np.std(start_times) if len(start_times) > 1 else 0

        # notes per beat
        notes_per_beat = num_notes / midi_obj.ticks_per_beat if midi_obj.ticks_per_beat > 0 else 0

        
        # combined feature vector
        features = [average_pitch, std_pitch, average_duration,
                    std_duration, note_density, pitch_range] + hist + hist_intervals + [
                        unique_pitches, articulation_rate, rest_ratio, 
                        polyphony_ratio, unique_durations, longest_rest,
                        onset_std, notes_per_beat]
        
        return features
    
    def predict(self, path, outpath=None):
        d = eval(open(path, 'r').read())
        predictions = {}
        
        for k in d:
            x = self.features(k)
            x_scaled = self.scaler.transform([x]) # scale the test features
            pred = self.model.predict(x_scaled)
            predictions[k] = str(pred[0])
            
        if outpath:
            with open(outpath, "w") as z:
                z.write(str(predictions) + '\n')
        return predictions

    # Train your model. Note that this function will not be called from the autograder:
    # instead you should upload your saved model using save()
    def train(self, path):
        with open(path, 'r') as f:
            train_json = eval(f.read())
            
        X_train = [self.features(k) for k in train_json]
        y_train = [train_json[k] for k in train_json]

        # normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # model = LogisticRegression(max_iter=1000)
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        self.model = model
        self.scaler = scaler


# ## Task 2: Sequence prediction

# In[198]:


dataroot2 = "student_files/task2_next_sequence_prediction/"


# In[428]:


class model2():
    def __init__(self):
        pass

    def get_first_last_pitch(self, path):
        filepath = path if os.path.exists(path) else os.path.join(dataroot2, path)
        notes = miditoolkit.midi.parser.MidiFile(filepath).instruments[0].notes

        # print("loading Midi: ", path)
        
        if not notes:
            return 0,0

        sorted_notes = sorted(notes, key=lambda n: n.start)
        return sorted_notes[0].pitch, sorted_notes[-1].pitch

    def features(self, path):
        # midi_obj = miditoolkit.midi.parser.MidiFile(dataroot2 + '/' + path)
        filepath = path if os.path.exists(path) else os.path.join(dataroot2, path)
        midi_obj = miditoolkit.midi.parser.MidiFile(filepath)
        # print("loading Midi: ", path)
        
        notes = midi_obj.instruments[0].notes
        if not notes:
            return [0]*4

        
        num_notes = len(notes)
        pitches = [note.pitch for note in notes]
        durations = [note.end - note.start for note in notes]
        total_time = max(note.end for note in notes) - min(note.start for note in notes)

        # main features
        average_pitch = sum([note.pitch for note in notes]) / num_notes
        std_pitch = np.std(pitches)
        average_duration = sum(durations) / num_notes
        note_density = num_notes / total_time if total_time > 0 else 0
        pitch_range = max(pitches) - min(pitches)

        # rest ratio
        sorted_notes = sorted(notes, key=lambda n: n.start)
        rest_time = 0.0
        for i in range(1, len(sorted_notes)):
            gap = sorted_notes[i].start - sorted_notes[i-1].end
            if gap > 0:
                rest_time += gap
        rest_ratio = rest_time / total_time if total_time > 0 else 0

        # pitch class histogram
        pitch_classes = [p%12 for p in pitches]
        hist = [0] * 12
        for classes in pitch_classes:
            hist[classes] += 1 
        hist = [x / num_notes for x in hist]

        # interval histogram
        intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]
        hist_intervals = [0] * 5
        for interval in intervals:
            if interval < -6:
                hist_intervals[0] += 1
            elif interval < 0:
                hist_intervals[1] += 1
            elif interval == 0:
                hist_intervals[2] += 1
            elif interval <= 6:
                hist_intervals[3] += 1
            else:
                hist_intervals[4] += 1
        hist_intervals = [x / len(intervals) for x in hist_intervals] if intervals else [0] * 5
        
        
        features = [average_pitch, std_pitch, average_duration,
                    note_density, pitch_range, rest_ratio] + hist + hist_intervals
        return features
    
    def train(self, path):
        # This baseline doesn't use any model (it just measures feature similarity)
        # You can use this approach but *probably* you'll want to implement a model
        with open(path, 'r') as f:
            train_json = eval(f.read())

        X_train = []
        y_train = []

        for k in train_json:
            path1, path2 = k
            x1 = self.features(path1)
            x2 = self.features(path2)
            diff = [a - b for a,b in zip(x1, x2)]

            f1, l1 = self.get_first_last_pitch(path1)
            f2, l2 = self.get_first_last_pitch(path2)
            pitch_transition = [abs(f2 - l1)]

            rhythm_similarity = [abs(x1[2] - x2[2])]
            
            combined = x1 + x2 + diff + pitch_transition + rhythm_similarity
            X_train.append(combined)
            y_train.append(train_json[k])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        model.fit(X_train_scaled, y_train)

        self.model = model
        self.scaler = scaler

    def predict(self, path, outpath=None):
        d = eval(open(path, 'r').read())
        predictions = {}
        for k in d:
            path1,path2 = k # Keys are pairs of paths
            x1 = self.features(path1)
            x2 = self.features(path2)
            diff = [a - b for a,b in zip(x1,x2)]
            # pitch transition
            f1, l1 = self.get_first_last_pitch(path1)
            f2, l2 = self.get_first_last_pitch(path2)
            pitch_transition = [f2 - l1]
            rhythm_similarity = [abs(x1[2] - x2[2])]
            combined = x1 + x2 + diff + pitch_transition + rhythm_similarity
            
            x_scaled = self.scaler.transform([combined])
            pred = self.model.predict(x_scaled)
            predictions[k] = bool(pred[0])
            
            # # Note: hardcoded difference between features
            # if abs(x1[0] - x2[0]) < 5:
            #     predictions[k] = True
            # else:
            #     predictions[k] = False
        
        if outpath:
            with open(outpath, "w") as z:
                z.write(str(predictions) + '\n')
        return predictions


# ## Task 3: Audio classification

# In[203]:


# Some constants (you can change any of these if useful)
SAMPLE_RATE = 16000
N_MELS = 64
N_CLASSES = 10
AUDIO_DURATION = 10 # seconds
BATCH_SIZE = 32


# In[205]:


dataroot3 = "student_files/task3_audio_classification/"


# In[207]:


def extract_waveform(path):
    waveform, sr = librosa.load(dataroot3 + '/' + path, sr=SAMPLE_RATE)
    waveform = np.array([waveform])
    if sr != SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resample(waveform)
    # Pad so that everything is the right length
    target_len = SAMPLE_RATE * AUDIO_DURATION
    if waveform.shape[1] < target_len:
        pad_len = target_len - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :target_len]
    waveform = torch.FloatTensor(waveform)
    return waveform


# In[273]:


class AudioDataset(Dataset):
    def __init__(self, meta, preload = True):
        self.meta = meta
        ks = list(meta.keys())
        self.idToPath = dict(zip(range(len(ks)), ks))
        self.pathToFeat = {}

        self.mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)
        self.db = AmplitudeToDB()
        
        self.preload = preload # Determines whether the features should be preloaded (uses more memory)
                               # or read from disk / computed each time (slow if your system is i/o-bound)
        if self.preload:
            for path in ks:
                waveform = extract_waveform(path)
                mel_spec = self.db(self.mel(waveform)).squeeze(0)
                self.pathToFeat[path] = mel_spec

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        # Faster version, preloads the features
        path = self.idToPath[idx]
        tags = self.meta[path]
        bin_label = torch.tensor([1 if tag in tags else 0 for tag in TAGS], dtype=torch.float32)

        if self.preload:
            mel_spec = self.pathToFeat[path]
        else:
            waveform = extract_waveform(path)
            mel_spec = self.db(self.mel(waveform)).squeeze(0)
        
        return mel_spec.unsqueeze(0), bin_label, path


# In[211]:


class Loaders():
    def __init__(self, train_path, test_path, split_ratio=0.9, seed = 0):
        torch.manual_seed(seed)
        random.seed(seed)
        
        meta_train = eval(open(train_path, 'r').read())
        l_test = eval(open(test_path, 'r').read())
        meta_test = dict([(x,[]) for x in l_test]) # Need a dictionary for the above class
        
        all_train = AudioDataset(meta_train)
        test_set = AudioDataset(meta_test)
        
        # Split all_train into train + valid
        total_len = len(all_train)
        train_len = int(total_len * split_ratio)
        valid_len = total_len - train_len
        train_set, valid_set = random_split(all_train, [train_len, valid_len])
        
        self.loaderTrain = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.loaderValid = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.loaderTest = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# In[213]:


class CNNClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * (N_MELS // 4) * (801 // 4), 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, mel/2, time/2)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, mel/4, time/4)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))  # multilabel → sigmoid


# In[215]:


class Pipeline():
    def __init__(self, model, learning_rate, seed = 0):
        # These two lines will (mostly) make things deterministic.
        # You're welcome to modify them to try to get a better solution.
        torch.manual_seed(seed)
        random.seed(seed)

        self.device = torch.device("cpu") # Can change this if you have a GPU, but the autograder will use CPU
        self.model = model.to(self.device) #model.cuda() # Also uncomment these lines for GPU
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

    def evaluate(self, loader, threshold=0.5, outpath=None):
        self.model.eval()
        preds, targets, paths = [], [], []
        with torch.no_grad():
            for x, y, ps in loader:
                x = x.to(self.device) #x.cuda()
                y = y.to(self.device) #y.cuda()
                outputs = self.model(x)
                preds.append(outputs.cpu())
                targets.append(y.cpu())
                paths += list(ps)
        
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        preds_bin = (preds > threshold).float()
        
        predictions = {}
        for i in range(preds_bin.shape[0]):
            predictions[paths[i]] = [TAGS[j] for j in range(len(preds_bin[i])) if preds_bin[i][j]]
        
        mAP = None
        if outpath: # Save predictions
            with open(outpath, "w") as z:
                z.write(str(predictions) + '\n')
        else: # Only compute accuracy if we're *not* saving predictions, since we can't compute test accuracy
            mAP = average_precision_score(targets, preds, average='macro')
        return predictions, mAP

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for x, y, path in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x = x.to(self.device) #x.cuda()
                y = y.to(self.device) #y.cuda()
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            val_predictions, mAP = self.evaluate(val_loader)
            print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f} | Val mAP: {mAP:.4f}")


# ## Run everything...

# In[380]:


def run1():
    model = model1()
    model.train(dataroot1 + "/train.json")
    train_preds = model.predict(dataroot1 + "/train.json")
    test_preds = model.predict(dataroot1 + "/test.json", "predictions1.json")
    
    train_labels = eval(open(dataroot1 + "/train.json").read())
    acc1 = accuracy1(train_labels, train_preds)
    print("Task 1 training accuracy = " + str(acc1))


# In[429]:


def run2():
    model = model2()
    model.train(dataroot2 + "/train.json")
    train_preds = model.predict(dataroot2 + "/train.json")
    test_preds = model.predict(dataroot2 + "/test.json", "predictions2.json")
    
    train_labels = eval(open(dataroot2 + "/train.json").read())
    acc2 = accuracy2(train_labels, train_preds)
    print("Task 2 training accuracy = " + str(acc2))


# In[222]:


def run3():
    loaders = Loaders(dataroot3 + "/train.json", dataroot3 + "/test.json")
    model = CNNClassifier()
    pipeline = Pipeline(model, 1e-4)
    
    pipeline.train(loaders.loaderTrain, loaders.loaderValid, 5)
    train_preds, train_mAP = pipeline.evaluate(loaders.loaderTrain, 0.5)
    valid_preds, valid_mAP = pipeline.evaluate(loaders.loaderValid, 0.5)
    test_preds, _ = pipeline.evaluate(loaders.loaderTest, 0.5, "predictions3.json")
    
    all_train = eval(open(dataroot3 + "/train.json").read())
    for k in valid_preds:
        # We split our training set into train+valid
        # so need to remove validation instances from the training set for evaluation
        all_train.pop(k)
    acc3 = accuracy3(all_train, train_preds)
    print("Task 3 training mAP = " + str(acc3))


# In[336]:


run1()


# In[ ]:


run2()


# In[68]:


run3()


# In[ ]:




