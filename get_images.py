# -*- coding: utf-8 -*-
import pandas as pd
import os

p = '/home/g1g/Desktop/fariha/NC2016_Test0601/reference/manipulation'
manup = pd.read_csv(p+"/NC2016-manipulation-ref.csv", sep='|', encoding='utf-8')

selected = manup[[
       'ProbeFileName',
       'ProbeMaskFileName',
       'IsTarget',
       'BaseFileName'
       ]]


basep1 = '/home/g1g/Desktop/fariha/NC2016_Test0601/'
d = []

for i, row  in selected.iterrows():
    if row.IsTarget == 'N':
        prob = basep1+row.ProbeFileName
        if os.path.isfile(prob):
            d.append({'BaseFileName' :row.BaseFileName,
                      'IsTarget': row.IsTarget,
                      'ProbeFileName': row.ProbeFileName, 
                      'ProbeMaskFileName': row.ProbeMaskFileName,
                      })
    elif row.IsTarget == 'Y':
        prob = basep1+row.ProbeFileName
        world = basep1+row.BaseFileName
        mask = basep1+row.ProbeMaskFileName
        if os.path.isfile(prob) and os.path.isfile(world) and os.path.isfile(mask):
            d.append({'BaseFileName' :row.BaseFileName,
                      'IsTarget': row.IsTarget,
                      'ProbeFileName': row.ProbeFileName, 
                      'ProbeMaskFileName': row.ProbeMaskFileName,
                      })
        
dd = pd.DataFrame(d)
df = dd.reindex(columns=['ProbeFileName','ProbeMaskFileName','IsTarget','BaseFileName'])
df.to_csv("newManupulation.csv", sep='\t', encoding='utf-8', index=False)
#        mask = basep1+row.row.ProbeMaskFileName
#        world = basep1+row.BaseFileName
    