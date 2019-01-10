
import os
import numpy as np
import pandas as pd
import dataset


folder='C:\\Kaggle\\atlas\\submissions'
submission_col = ['Id', 'Predicted']

LABEL_ALL = list(map(str,range(28)))


def find_inputs(folder, types=['.csv']):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                inputs.append((base, abs_filename))
    return inputs


def vector_to_tags(v, tags):
    idx = np.nonzero(v)
    t = [tags[i] for i in idx[0]]
    return ' '.join(t)


def main():

    subs = find_inputs(folder, types=['.csv'])
    dfs = []
    for s in subs:
        df = pd.read_csv(s[1], index_col=None)
        df = df.set_index('Id')
        df.Predicted = df.Predicted.map(lambda x: set(str(x).split()))
        for l in LABEL_ALL:
            df[l] = [1 if l in tag else 0 for tag in df.Predicted]
        df.drop(['Predicted'], inplace=True, axis=1)
        dfs.append(df)

    assert len(dfs)
    d = dfs[0]
    for o in dfs[1:]:
        d = d.add(o)
    d = d / len(dfs)
    b = (d > 0.5).astype(int)

    tags = dataset.LABEL_ALL
    m = b.as_matrix()
    out = []
    for i, x in enumerate(m):
        t = vector_to_tags(x, tags)
        out.append([b.index[i]] + [t])

    results_sub_df = pd.DataFrame(out, columns=submission_col)
    results_sub_df.to_csv(os.path.join(folder,'submission-e.csv'), index=False)


if __name__ == '__main__':
    main()
