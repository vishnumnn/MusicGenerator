import music_gen
from sklearn.model_selection import train_test_split

## File lists
PATHS = ['Fantasie_Impromptu.mid',
         'Moonlight_Sonata_3rd_Movement_-_Ludwig_van_Beethoven.mxl', 'Paganini_Caprice_No_5_in_A_minor.mxl',
        'Liszt_Romance_S._169.mxl', 'Musical_Moment.mxl', "Mephisto_Waltz_No._1_S._514.mxl"]
## Includes vitali chaconne in addition to paths
PATHS_2 = ['Solo_Violin_Sonata_No._1_in_G_Minor_-_J._S._Bach_BWV_1001.mxl',
         'Moonlight_Sonata_3rd_Movement_-_Ludwig_van_Beethoven.mxl', 'Paganini_Caprice_No_5_in_A_minor.mxl',
          'Vitali_Chaconne_Solo_Violin.mxl', 'Liszt_Romance_S._169.mxl','Fantasie_Impromptu.mxl']

### remove
PATHS_3 = ['Vitali_Chaconne_Solo_Violin.mxl']

# Clean data
Seqs, Labels = music_gen.cleanData(PATHS_3, 50)
###

# # Clean data
# Seqs, Labels = music_gen.cleanData(PATHS, 50)
# print('''Overall shape of sequences data {0}'''.format(Seqs.shape))

# # Split training and test data
# Train_Sequences, Test_Sequences, Train_Labels, Test_Labels = train_test_split(Seqs, Labels, test_size = 0.004, random_state = 56)

# Create and Train model (saves model and weights to disk)
m_filepath, w_filepath = music_gen.create_and_train_model_V2(PATHS)


# Pick a random seed sequence to generate predictions (could test multiple seed prediction in future)
predictions = music_gen.predict_with_saved_weights_V2(m_filepath, w_fliepath, Seqs[167], 400)

# Write prediction in midi format
music_gen.create_MIDI_file_multilabel(predictions)