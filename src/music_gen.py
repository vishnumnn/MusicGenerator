## Imports 1
import numpy as np
import keras
import tensorflow as tf
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

## CONSTANTS
_DATE_TIME_FORMAT = "%d_%m_%Y_%H_%M_%S"
_DATETIME = date_and_time = datetime.now().strftime(_DATE_TIME_FORMAT)
_CHORD_MULTIPLIER = 0.5

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
        dur = duration.Duration(quarterLength=4.0)
        if(i < arrayLen - 1):
            dur.quarterLength = tupleArray[i + 1][1] - offset
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
        element.duration = d
        s.append(element)
    return s

## Convert note-dur list to midi only multi label encoding
def noteToMidiNumbers(nList):
    # 88 to represent 88 midi encodings and 1 for rest
    data = np.zeros((len(nList), 102))
    for i in range(len(nList)):
        if(nList[i].isRest):
            data[i,101] = 1
        else:
            pitches = nList[i].pitches
            for e in pitches:
                data[i,e.midi] = 1
                ## Comment the break IF YOU WANT TO ENCODE ALL NOTES IN A CHORD NOT JUST THE FIRST
                # break
    return data

def getData(score):
        intermediate = notesAndRests(score)
        intermediate = pitchesAndOffsetTuples(intermediate)
        intermediate = groupPitchesByOffset(intermediate) 
        intermediate = reconstructListOfNotesAndDurations(intermediate)
        intermediate = noteToMidiNumbers(intermediate)
        print('''Number of notes: {0}'''.format(intermediate.shape[0]))
        return intermediate

## Group Multi-Label Encodings into Sequences and Corresponding Labels

## Consider altering function so that sequences can be found at halfway points between labels recursively up to a 
## certain depth. E.g. Sequences at every 0th offset, Seqlen/2 offset, SeqLen/4 offset, and so on. 
def getSeqsAndLabelsForSingleScore(data, SeqLen):
    ## data is a 2d numpy array, SeqLen is an integer
    numSeqs = math.floor(data.shape[0]/(SeqLen + 1))
    ## Numpy array of Seqs
    bridgeAddition = math.floor(numSeqs - math.floor(SeqLen/2) / SeqLen)
    SeqSet = np.zeros((numSeqs + bridgeAddition, SeqLen, data.shape[1]))
    ## Numpy array of Labels
    SeqLabels = np.zeros((numSeqs + bridgeAddition, data.shape[1]))
    for i in range(numSeqs - 1):
        SeqSet[i] = data[i*SeqLen : (i+1)*SeqLen]
        SeqLabels[i] = data[(i+1)*SeqLen]
    offset = math.floor(SeqLen/2)
    for i in range(numSeqs, numSeqs + bridgeAddition - 1):
        multiple = i - numSeqs
        SeqSet[i] = data[offset + multiple*SeqLen : offset + (multiple + 1)*SeqLen]
        SeqLabels[i] = data[offset + (multiple + 1)*SeqLen]
    return (SeqSet, SeqLabels)

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
def getSeqsAndLabels(scores, SeqLen):
    SeqSet, SeqLabels = getSeqsAndLabelsPermutations(getData(scores.pop(0)), SeqLen)
    print(SeqSet.shape, SeqLabels.shape)
    for each in scores:
        D, L = getSeqsAndLabelsPermutations(getData(each), SeqLen)
        SeqSet = np.concatenate((SeqSet, D))
        SeqLabels = np.concatenate((SeqLabels, L))
        print(SeqSet.shape, SeqLabels.shape)
    return (SeqSet, SeqLabels)

def cleanData(filepaths, sequenceLength):
    # Load Files and Extract streams
    direc = os.getcwd()
    scores = list(map(lambda x: converter.parse(direc + '''/music/''' + x).parts.stream(), filepaths))
    return getSeqsAndLabels(scores, sequenceLength)

## Create the model and train it on the passed data. 
## Write the model and weights to a json and hdf5 file respectively.
def create_and_train_model(Seqs, Labels):
    ## Train Model
    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(Seqs.shape[1], Seqs.shape[2]),
        return_sequences=True
    ))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(Seqs.shape[2], activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    ## Train on everything except the first 10 samples
    model.fit(Seqs, Labels, epochs = 60, batch_size = 32)

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

## Creates and trains model on multiple labels on multiple categories
def create_and_train_model_V2(Seqs, Labels):
    ## Train Model
    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(Seqs.shape[1], Seqs.shape[2]),
        return_sequences=True
    ))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(Seqs.shape[2], activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    ## Train on everything except the first 10 samples
    model.fit(Seqs, Labels, epochs = 80, batch_size = 32)

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

def predict_with_saved_weights(json_path, h5_path, seed_data, number_of_notes):
    ## seed_data is a 2 dimensional input (sequence of one-hot-encoded notes)
    # Load model
    json_file = open(json_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    # Load weights into model
    model.load_weights(h5_path)

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    inp = seed_data.tolist()
    predictions = []
    i = 0
    while(i < number_of_notes):
        inpNP = np.asarray(inp)
        pred = model.predict(np.reshape(inpNP, (1,inpNP.shape[0],inpNP.shape[1])))
        ## Currently only chooses the maximum of the predicted array for storage
        inp.append(pred[0])
        draw = np.random.choice(np.arange(0,inpNP.shape[1]),p=pred[0], replace = True)
        predictions.append(draw)
        inp = inp[1:len(inp)]
        i += 1        
    return predictions

## TODO: Test changing added note to what is there after choosing cutoff is applied
def predict_with_saved_weights_V2(json_path, h5_path, seed_data, number_of_notes):
    ## seed_data is a 2 dimensional input (sequence of one-hot-encoded notes)
    # Load model
    json_file = open(json_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    # Load weights into model
    model.load_weights(h5_path)

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    inp = seed_data.tolist()
    predictions = []
    i = 0
    while(i < number_of_notes):
        inpNP = np.asarray(inp)
        pred = model.predict(np.reshape(inpNP, (1,inpNP.shape[0],inpNP.shape[1])))
        # Currently only chooses the maximum of the predicted array for storage
        inp.append(pred[0])
        ## TODO: Take the highest value. Remove the notes above and below it. Then choose all the notes that are within 0.15*(it's probability)
        ## points
        prediction = pred[0]

        draw = np.where(prediction == np.amax(prediction))[0][0]
        draw_prob = prediction[draw]

        args_to_remove = [draw, draw - 1, draw + 1]
        prediction = np.delete(prediction, args_to_remove)

        chord_notes = np.where(prediction > _CHORD_MULTIPLIER*draw_prob)[0]
        print(chord_notes)
        draw = np.append(chord_notes, draw)

        predictions.append(draw)
        inp = inp[1:len(inp)]
        i += 1        
    return predictions

def create_MIDI_file(predicted_notes):
    s = stream.Stream()
    for m in predicted_notes:
        if(m == 101):
            n = note.Rest()
        else:
            p = pitch.Pitch(m)
            n = note.Note()
            n.pitch = p
        n.duration = duration.Duration(quarterLength = 1)
        s.append(n)
    # speed up piece by factor of 2
    stream_to_write = s.augmentOrDiminish(0.50)
    # write to midi file
    MIDI_filepath = os.getcwd() + '''/output/music_gen_output_{0}.mid'''.format(_DATETIME)
    stream_to_write.write('midi', fp= MIDI_filepath)

def create_MIDI_file_multilabel(predicted_notes):
    s = stream.Stream()
    for m in predicted_notes:
        arr = m[np.where(m != 101)]
        if(arr.size == 0):
            s.append(note.Rest())
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
    stream_to_write = s.augmentOrDiminish(0.50)
    # write to midi file
    MIDI_filepath = os.getcwd() + '''/output/music_gen_output_{0}.mid'''.format(_DATETIME)
    stream_to_write.write('midi', fp= MIDI_filepath)