import music_gen
from sklearn.model_selection import train_test_split

## File lists
PATHS = ['Fantasie_Impromptu.mid',
         'Moonlight_Sonata_3rd_Movement_-_Ludwig_van_Beethoven.mxl', 'Paganini_Caprice_No_5_in_A_minor.mxl',
        'Liszt_Romance_S._169.mxl']
## Includes vitali chaconne in addition to paths
PATHS_2 = ['Solo_Violin_Sonata_No._1_in_G_Minor_-_J._S._Bach_BWV_1001.mxl',
         'Moonlight_Sonata_3rd_Movement_-_Ludwig_van_Beethoven.mxl', 'Paganini_Caprice_No_5_in_A_minor.mxl',
          'Vitali_Chaconne_Solo_Violin.mxl', 'Liszt_Romance_S._169.mxl','Fantasie_Impromptu.mxl']

# Clean data
Seqs, Labels = music_gen.cleanData(PATHS, 50)
print('''Overall shape of sequences data {0}'''.format(Seqs.shape))

# Split training and test data
Train_Sequences, Test_Sequences, Train_Labels, Test_Labels = train_test_split(Seqs, Labels, test_size = 0.004, random_state = 17)

# Create and Train model (saves model and weights to disk)
m_filepath, w_fliepath = music_gen.create_and_train_data(Train_Sequences, Train_Labels)

# Pick a random seed sequence to generate predictions (could test multiple seed prediction in future)
predictions = music_gen.predict_with_saved_weights(m_filepath, w_fliepath, Test_Sequences[5], 50)

# Write prediction in midi format
music_gen.create_MIDI_file(predictions)
