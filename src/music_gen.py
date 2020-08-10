## Imports 1
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.utils import plot_model
from keras.layers import Activation
from music21 import *
import math, os
from datetime import datetime
from numpy.random import choice
import gc

## PARAMETER CONSTANTS
_DATE_TIME_FORMAT = "%d_%m_%Y_%H_%M_%S"
_DATETIME = date_and_time = datetime.now().strftime(_DATE_TIME_FORMAT)
_CHORD_MULTIPLIER = 0.5
_NOTE_CATS = 106
_BATCH_SIZE = 32
_EPOCHS = 160
_LSTM_NODE_COUNT = 512
_TICS_PER_MEASURE = 48
_SEQUENCE_LENGTH = _TICS_PER_MEASURE*2 #number of quarter notes in a sequence. 12 tics per quarter note in 4/4

callbacks = [
# This callback saves a SavedModel every x batches
keras.callbacks.ModelCheckpoint(
    filepath= os.getcwd() + '/models/chkpts/ckpt-acc={accuracy:.2f}',
    save_freq= 10000)
]
## Get notes and rests per instrument from score
def notesAndRests(score):
    instruments = instrument.partitionByInstrument(score)
    noteMatrix = []
    i = instruments[0]
    for NoteRestChord in i.notesAndRests:
        noteMatrix.append(NoteRestChord)
    return noteMatrix

## Replace noteMatrix with matrix containing tuples of pitch-offset information
## Pitches are used from this point on to identify recreate chords based on offset information because some notes,
## though in the same chord, can have varying durations. Note offsets in chords can also have discrepancies based
## on the file's condition. Pairing pitches to the offset of the chord they originate from avoids this. 
def pitchesAndOffsetTuples(score):
    for i in range(len(score)):
        element = score[i]
        pitchInfo = [element]
        if(element.isChord):
            pitchInfo = list(element.pitches)
        elif(element.isNote):
            pitchInfo = [element.pitch]
        score[i] = (pitchInfo, element.offset)
    return score

## Group pitches occuring at same offset into pitch-duration tuples
## Reconstruct the duration of a set of pitches to be added to the regrouped chords and notes
def groupPitchesByOffset(tupleArray):
    pitchesAndDuration = []
    arrayLen = len(tupleArray)
    i = 0
    while(i < arrayLen):
        pitches,offset = tupleArray[i]
        while(i + 1 < arrayLen and tupleArray[i + 1][1] == offset):
            i += 1
            # Add all of the pitches in the tuple with the same offset as tuple i to this offset's group of pitches
            if(len(tupleArray[i][0]) > 1  or type(tupleArray[i][0][0]) != type(note.Rest())):
                pitches.extend(tupleArray[i][0])
        dur = 1.0
        if(i < arrayLen - 1):
            dur = tupleArray[i + 1][1] - offset
        if(type(pitches[0]) == type(note.Rest()) and len(pitches) > 1):
            pitches.pop(0)
        pitchesAndDuration.append((pitches,dur))
        i += 1
    return pitchesAndDuration

## Reconstruct notes and chords from the pitches and durations, used to test if the data is still faithful
## to the original piece. 
def reconstructListOfNotesAndDurations(tuplesArray):
    ## [] can be replaced by stream.Stream to create a stream instead of a list
    s = []
    durs = []
    for each in tuplesArray:
        pitches, d = each
        if(len(pitches) == 1 and type(pitches[0]) == type(note.Rest())):
            element = pitches[0]
        else:
            pitchNames = list(map(lambda x: x.nameWithOctave, pitches))
            if(len(pitchNames) > 1):
                element = chord.Chord(pitchNames)
            else:
                element = note.Note(pitchNames[0])
        durs.append(d)
        s.append(element)
    return (s, durs)

## Convert note-dur list to midi only multi label encoding
def noteToMidiNumbers(nList, durList):
    # 107 to represent 105 midi encodings, 1 for rest, 1 for duration in decimal
    data = np.zeros((len(nList), _NOTE_CATS))
    max_dur = max(durList)
    for i in range(len(nList)):
        if(nList[i].isRest):
            data[i,_NOTE_CATS - 2] = 1
        else:
            pitches = nList[i].pitches
            for e in pitches:
                data[i,e.midi] = 1
                ## Comment the break IF YOU WANT TO ENCODE ALL NOTES IN A CHORD NOT JUST THE FIRST
                # break
        data[i, _NOTE_CATS - 1] = durList[i]/max_dur
    return data

def getData(score):
        intermediate = notesAndRests(score)
        intermediate = pitchesAndOffsetTuples(intermediate)
        intermediate = groupPitchesByOffset(intermediate) 
        intermediate = reconstructListOfNotesAndDurations(intermediate)
        intermediate = noteToMidiNumbers(intermediate[0], intermediate[1])
        return intermediate


def el_end(el):
    return el.offset + el.duration.quarterLength

def encode_element_array(elem_ar):
    encoded_ar = np.zeros((_NOTE_CATS,))
    for e in elem_ar:
        if(e.isRest):
            encoded_ar[_NOTE_CATS - 1] = 1
        else:        
            pitches = e.pitches
            for p in pitches:
                encoded_ar[p.midi] = 1
    return encoded_ar
        
def encode_all_elements(elems):
    window_start = elems[0].offset
    window_end = elems[0].offset
    
    elems_idx = 0
    encoded_data_idx = 0
    
    note_store = [elems[0]]
    shortest_note_in_store = elems[0]
    final_elem = elems[len(elems) - 1]
    encoded_data = np.ndarray((int(round(el_end(final_elem) * _TICS_PER_MEASURE)), _NOTE_CATS))
    
    def process_window(y):
        nonlocal encoded_data_idx
        clean = encode_element_array(note_store)
        span = int(round((window_end - window_start)* _TICS_PER_MEASURE))
        for i in range(span):
            encoded_data[encoded_data_idx + i] = clean
        encoded_data_idx += span 
        
    print(el_end(final_elem))
    while(window_start < el_end(final_elem)):
        shortest_note_end = el_end(shortest_note_in_store)
        if(elems_idx + 1 < len(elems)):
            next_elem = elems[elems_idx + 1]
            if(next_elem.duration.quarterLength == 0.0):
                elems_idx += 1
                continue
            if(next_elem.offset < shortest_note_end):
                if(next_elem.offset > shortest_note_in_store.offset):
                    window_end = next_elem.offset
                    process_window()
                    window_start = window_end
                note_store.append(next_elem)
                shortest_note_in_store = min(note_store, key = lambda x: el_end(x))
                elems_idx += 1
                continue
        window_end = shortest_note_end
        process_window()
        note_store = [note for note in note_store if el_end(note) > window_end]
        if(not note_store):
            note_store.append(next_elem)
        window_start = window_end
        shortest_note_in_store = min(note_store, key = lambda x: el_end(x))
    return encoded_data

## Every increasing permutation instead of every half sequence len
def getSeqsAndLabelsPermutations(data, SeqLen):
    ## data is a 2d numpy array, SeqLen is an integer
    numSeqs = math.floor(data.shape[0] - SeqLen)
    ## Numpy array of Seqs
    SeqSet = np.zeros((numSeqs, SeqLen, data.shape[1]))
    ## Numpy array of Labels
    SeqLabels = np.zeros((numSeqs, data.shape[1]))
    for i in range(numSeqs):
        SeqSet[i] = data[i : i + SeqLen]
        SeqLabels[i] = data[i + SeqLen]
    return (SeqSet, SeqLabels)

## Outputs a ndarray of (Num Sequences, Sequence Length, Num features) schema
def getSeqsAndLabels(sequences_labels_tuples, SeqLen):
    SeqSet, SeqLabels = sequences_labels_tuples[0]
    for i in range(1, len(sequences_labels_tuples)):
        D, L = sequences_labels_tuples[i]
        SeqSet = np.concatenate((SeqSet, D))
        SeqLabels = np.concatenate((SeqLabels, L))
    return (SeqSet, SeqLabels)

def clean_notewise_data(filepaths, sequence_length):
    # Load Files and Extract streams
    scores = list(map(lambda x: converter.parse(os.getcwd() + '''/music/''' + x).parts.stream(), filepaths))
    encoded_data = list(map(getData, scores))
    sequenced_data = list(map(lambda x: getSeqsAndLabelsPermutations(x, sequence_length), encoded_data))
    del scores
    del encoded_data
    del sequenced_data
    gc.collect()
    SeqSet, SeqLabels = getSeqsAndLabels(sequenced_data, sequence_length)
    return (SeqSet, SeqLabels)

def clean_ticwise_data(filepaths, sequence_length):
    # Load Files and Extract streams
    scores = list(map(lambda x: converter.parse(os.getcwd() + '''/music/''' + x).parts.stream(), filepaths))
    notes_for_scores = list(map(notesAndRests, scores))
    encoded_data = list(map(encode_all_elements, notes_for_scores))
    sequenced_data = list(map(lambda x: getSeqsAndLabelsPermutations(x, sequence_length), encoded_data))    
    del scores
    del notes_for_scores
    del encoded_data
    del sequenced_data
    gc.collect()
    SeqSet, SeqLabels = getSeqsAndLabels(sequenced_data, sequence_length)
    return (SeqSet, SeqLabels)

# TODO: Experiment with deleting and recreating model each epoch to prevent memory leaks
# TODO: Factor in duration as a new feature
## Create and Train model without using generator to feed data.
def create_and_train_model(Seqs, Labels, Use_Checkpoint = False):
    # Train Model
    model = None
    if(Use_Checkpoint):
        model = restore_model_from_checkpoints()
    if(model == None):
        print("Nodes: ",_LSTM_NODE_COUNT, "Sequence length: ", _SEQUENCE_LENGTH)
        model = Sequential()
        model.add(LSTM(
            _LSTM_NODE_COUNT,
            input_shape=(_SEQUENCE_LENGTH, _NOTE_CATS),
            return_sequences=True
        ))
        model.add(LSTM(_LSTM_NODE_COUNT, return_sequences=True))
        model.add(LSTM(_LSTM_NODE_COUNT))
        model.add(Dense(_NOTE_CATS, activation = 'sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(Seqs, Labels, epochs = _EPOCHS, batch_size = _BATCH_SIZE, callbacks = callbacks)

    # serialize model to JSON
    model_json = model.to_json()
    
    JSON_filepath = os.getcwd() + '''/models/model_{0}.json'''.format(_DATETIME)
    HDF5_filepath = os.getcwd() + '''/models/weights_{0}.h5'''.format(_DATETIME)

    # write model to json
    with open(JSON_filepath, "w") as json_file:
        json_file.write(model_json)
    print("Wrote model to JSON")

    # serialize weights to HDF5
    model.save_weights(HDF5_filepath)
    print("Saved weights to hdf5")

    return (JSON_filepath, HDF5_filepath)

## Cleaned data generator
def train_batch_generator(scores, mode):
    # BE WARY OF MODE BEING EVAL
    # input is the set of scores
    
    score_counter = 0
    Seqs, Labels = getSeqsAndLabels([scores[score_counter]], _SEQUENCE_LENGTH)
    a = np.arange(Seqs.shape[0])
    remaining_data_count = 0
    while(True):
        indices = np.random.choice(a.size, _BATCH_SIZE - remaining_data_count, replace=False)
        batch = Seqs[indices, :]
        Labels = Seqs[indices, :]
        a = np.delete(a, indices)
        if(a.size <= _BATCH_SIZE):
            remaining_data_count = a.size
            score_counter = 0 if (score_counter + 1) >= len(scores) else (score_counter + 1)
            Seqs, Labels = getSeqsAndLabels([scores[score_counter]], _SEQUENCE_LENGTH)
            a = np.arange(Seqs.shape[0])
        yield batch, label
            
## Uses Generator to supply data. 
def create_and_train_model_V2(paths, Use_Checkpoint = False):
    # Train 
    model = None
    if(Use_Checkpoint):
        model = restore_model_from_checkpoints()
    if(model == None):
        model = Sequential()
        model.add(LSTM(
            256,
            input_shape=(_SEQUENCE_LENGTH, _NOTE_CATS),
            return_sequences=True
        ))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(256))
        model.add(Dense(_NOTE_CATS, activation = 'sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Set up generator
    scores = list(map(lambda x: converter.parse(os.getcwd() + '''/music/''' + x).parts.stream(), paths))
    batch_generator = train_batch_generator(scores, "train")
    model.fit(x = batch_generator, shuffle = True, epochs = _EPOCHS, steps_per_epoch = _DATA_COUNT/_BATCH_SIZE, callbacks = callbacks)
    # serialize model to JSON
    model_json = model.to_json()
    
    JSON_filepath = os.getcwd() + '''/models/model_{0}.json'''.format(_DATETIME)
    HDF5_filepath = os.getcwd() + '''/models/weights_{0}.h5'''.format(_DATETIME)

    # write model to json
    with open(JSON_filepath, "w") as json_file:
        json_file.write(model_json)
    print("Wrote model to JSON")

    # serialize weights to HDF5
    model.save_weights(HDF5_filepath)
    print("Saved weights to hdf5")

    return (JSON_filepath, HDF5_filepath)

def predict_with_saved_weights_checkpoint(seed_data, number_of_notes):
    model = restore_model_from_checkpoints()
    inp = seed_data.tolist()
    predictions = []
    i = 0
    while(i < number_of_notes):
        inpNP = np.asarray(inp)
        pred = model.predict(np.reshape(inpNP, (1,inpNP.shape[0],inpNP.shape[1])))
        # Currently only chooses the maximum of the predicted array for storage
        inp.append(pred[0])
        prediction = pred[0]

        draw = np.where(prediction == np.amax(prediction))[0][0]
        draw_prob = prediction[draw]

        args_to_remove = [draw, draw - 1, draw + 1]
        prediction = np.delete(prediction, args_to_remove)

        chord_notes = np.where(prediction > _CHORD_MULTIPLIER*draw_prob)[0]
        draw = np.append(chord_notes, draw)

        predictions.append((draw, duration_prediction))
        inp = inp[1:len(inp)]
        i += 1        
    return predictions

def predict_with_saved_weights_json(json_path, h5_path, seed_data, number_of_notes):
    ## seed_data is a 2 dimensional input (sequence of one-hot-encoded notes)
    # Load model
    model = restore_model_from_json(json_path, h5_path)
    inp = seed_data.tolist()
    predictions = []
    i = 0
    while(i < number_of_notes):
        inpNP = np.asarray(inp)
        pred = model.predict(np.reshape(inpNP, (1,inpNP.shape[0],inpNP.shape[1])))
        # Currently only chooses the maximum of the predicted array for storage
        inp.append(pred[0])
        prediction = pred[0]

        draw = np.where(prediction == np.amax(prediction))[0][0]
        draw_prob = prediction[draw]

        args_to_remove = [draw, draw - 1, draw + 1]
        prediction = np.delete(prediction, args_to_remove)

        chord_notes = np.where(prediction > _CHORD_MULTIPLIER*draw_prob)[0]
        draw = np.append(chord_notes, draw)

        predictions.append((draw, duration_prediction))
        inp = inp[1:len(inp)]
        i += 1        
    return predictions

def create_MIDI_file_multilabel(predicted_notes, tempo_scale):
    s = stream.Stream()

    for m,dur in predicted_notes:
        arr = m[np.where(m != _NOTE_CATS - 1)]
        if(arr.size == 0):
            note_to_add = note.Rest()
            s.append()
            continue
        if(arr.size == 1):
            n = note.Note()
            n.pitch = pitch.Pitch(arr[0])
            s.append(n)
            continue
        if(arr.size > 1):
            midis = arr.tolist()
            pitches = list(map(lambda x: pitch.Pitch(x), midis))
            c = chord.Chord(pitches)
            s.append(c)
        else:
            print("something went wrong m: {0} arr: {1}".format(m, arr))
    # speed up piece by factor of 2
    stream_to_write = s.augmentOrDiminish(tempo_scale)
    # write to midi file
    MIDI_filepath = os.getcwd() + '''/output/music_gen_output_{0}.mid'''.format(_DATETIME)
    stream_to_write.write('midi', fp= MIDI_filepath)


## MEMORY OPTIMIZATIONS
def restore_model_from_checkpoints():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    chkpt_dir = os.getcwd() + '/models/chkpts/'
    checkpoints = [chkpt_dir + name
                   for name in os.listdir(chkpt_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    return "no model found in checkpoints"

def restore_model_from_json(json_path, h5_path):
    json_file = open(json_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    # Load weights into model
    model.load_weights(h5_path)

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def find_smoothed_duration(dur):
    smallest_diff_index = 0 
    smallest_diff = abs(opt_array[0] - dur)
    for index, e in enumerate(opt_array, 1):
        diff = abs(dur - e)
        if(diff < smallest_diff):
            smallest_diff = diff
            smallest_diff_index = index
    return opt_array[smallest_diff_index]
