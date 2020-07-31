import pandas as pd
import music_gen

## File lists
PATHS = ['Fantasie_Impromptu.mid', 'Solo_Violin_Sonata_No._1_in_G_Minor_-_J._S._Bach_BWV_1001.mxl',
         'Moonlight_Sonata_3rd_Movement_-_Ludwig_van_Beethoven.mxl', 'Paganini_Caprice_No_5_in_A_minor.mxl',
        'Liszt_Romance_S._169.mxl']
## Includes vitali chaconne in addition to paths
PATHS_2 = ['Solo_Violin_Sonata_No._1_in_G_Minor_-_J._S._Bach_BWV_1001.mxl',
         'Moonlight_Sonata_3rd_Movement_-_Ludwig_van_Beethoven.mxl', 'Paganini_Caprice_No_5_in_A_minor.mxl',
          'Vitali_Chaconne_Solo_Violin.mxl', 'Liszt_Romance_S._169.mxl','Fantasie_Impromptu.mxl']

Seqs, Labels = music_gen.cleanData
print('''Overall shape of sequences data {0}'''.format(Seqs.shape))

# Create and Train call
#music_gen.createAndTrainData(Seqs, Labels)