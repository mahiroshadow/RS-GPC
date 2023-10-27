from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv('./dataset/user_artists.dat',
                 sep='\t',
                 names=['userID', 'itemID', 'weight'],
                 skiprows=1)
df_item = pd.read_csv('./dataset/artists.dat',
                      sep='\t',
                      names=['id', 'name', 'url', 'pictureURL'],
                      skiprows=1)
usr_set = set()
item_set = set()
for i in df['userID']:
    usr_set.add(i)

for i in df_item['id']:
    item_set.add(i)

usr_encoder = LabelEncoder().fit(list(usr_set))
item_encoder = LabelEncoder().fit(list(item_set))
