import pandas as pd
import json
import numpy as np
from collections import Counter
import math
import os

BASE_PATH = '/content/gdrive/My Drive'
TRAIN_CSV = 'train.csv'



    
def main():
    create_info_file()
    train_df=pd.read_csv(os.path.join(DRIVE,'train.csv'))
    train_df.columns=['Id','Target']
    train_df.Target = train_df.Target.map(lambda x: set(str(x).split()))
    count = Counter()
    train_df.Target.apply(lambda x: count.update(x))
    print(count)

    for k in count:
        train_df[k] = [1 if k in tag else 0 for tag in train_df.Target]

    attempt = 0
    num_folds = 10
    target_counts = {k: (v / num_folds) for k, v in count.items()}
    target_thresh = {k: max(1., v * .20) for k, v in target_counts.items()}

    print(target_counts, target_thresh)
    furthest_fold = 0
    fold_counts = []
 
    while attempt < 100:
        train_df['fold'] = np.random.randint(0, num_folds, size=len(train_df.index))
        valid = True
        ss = train_df.groupby('fold').sum()
        for f in range(num_folds):
            sr = ss.ix[f]
            fold_counts.append(sr)
            for k, v in sr.items():
                target = target_counts[k]
                thresh = target_thresh[k]
                diff = math.floor(abs(v - target))
                thresh = math.ceil(thresh)
                if diff > thresh:
                    valid = False
                    if f > furthest_fold:
                        furthest_fold = f
                        print(f, abs(v - target), math.ceil(thresh), k)
                    break
            if not valid:
                break
        if valid:
            break
        else:
            fold_counts = []
            attempt += 1
            
    print(attempt, furthest_fold)
    for i, x in enumerate(fold_counts):
        print(i)
        for k, v in x.items():
            print(k, v)
        print()
    
    labels_df = train_df[['Id', 'fold'] + sorted(list(count.keys()),key= lambda x: int(x))]
    labels_df.to_csv(os.path.join(DRIVE,'labels.csv'),index=False)

def create_info_file(): 
    with open('train.json') as json_data:
        data=json.load(json_data)
    ANNOTATIONS, CAT, INFO, LICENSES, IMAGES = data.keys()
    annotations = pd.DataFrame(data[ANNOTATIONS]).set_index('image_id')
    categories=pd.DataFrame(data[CAT])
    images=pd.DataFrame(data[IMAGES]).set_index('id')
    train_df=images.merge(annotations, 
                        left_index=True,
                        right_index=True, 
                        how='left').drop('id',axis=1).merge(categories,
                                                            left_on='category_id',
                                                            right_on='id',
                                                            how='left').drop(['id','license','rights_holder','supercategory','height','width','name'],axis=1)
    train_df.to_csv(os.path.join(DRIVE,'train.csv'),index=False)
    
if __name__ == '__main__':
    main()