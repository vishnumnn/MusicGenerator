## 1st arg is the json model path, 2nd is the weight path, 3rd is the number of notes to generate ##
import music_gen, sys

## File lists
PATHS = ['Fantasie_Impromptu.mid',
         'Moonlight_Sonata_3rd_Movement_-_Ludwig_van_Beethoven.mxl', 'Paganini_Caprice_No_5_in_A_minor.mxl',
        'Liszt_Romance_S._169.mxl']
## Includes vitali chaconne in addition to paths
PATHS_2 = ['Solo_Violin_Sonata_No._1_in_G_Minor_-_J._S._Bach_BWV_1001.mxl',
         'Moonlight_Sonata_3rd_Movement_-_Ludwig_van_Beethoven.mxl', 'Paganini_Caprice_No_5_in_A_minor.mxl',
          'Vitali_Chaconne_Solo_Violin.mxl', 'Liszt_Romance_S._169.mxl','Fantasie_Impromptu.mxl']

PATHS_3 = ['Vitali_Chaconne_Solo_Violin.mxl']

# Clean data
Seqs, Labels = music_gen.cleanData(PATHS_3, 50)
print('''Overall shape of sequences data {0}'''.format(Seqs.shape))

# Pick a random seed sequence to generate predictions (could test multiple seed prediction in future)
predictions = music_gen.predict_with_saved_weights(sys.argv[1], sys.argv[2], Seqs[25], int(sys.argv[3]))

# Write prediction in midi format
music_gen.create_MIDI_file(predictions)