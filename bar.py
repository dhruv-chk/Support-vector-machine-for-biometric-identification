# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:09:03 2018

@author: Dhruv
"""

import numpy as np
import matplotlib.pyplot as plt



N = 5
men_means = (97, 95, 96, 95, 5)
x=(1,2,3,4,5)
men_std = ("Walking", "Standing", "Sitting", "Gather", "Split")

ind = np.arange(N)  # the x locations for the groups
width = 0.5       # the width of the bars

fig, ax = plt.subplots()
#fig(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.ylim(0,130)
plt.xlim(0,7)
rects=ax.bar(x, men_means, width, align='center', alpha=0.85)
ax.set_ylabel('Accuracy')
ax.set_title('')
ax.set_xticks(x)
#ax.set_xticklabels(('Walking', 'Standing', 'Sitting', 'Gather', 'Split'), ha='center')
rects[0].set_label('Precision')
rects[1].set_label('Recall')
rects[2].set_label('F1 Score')
rects[3].set_label('Accuracy')
rects[4].set_label('Error')


rects[0].set_color('limegreen')
rects[1].set_color('firebrick')
rects[2].set_color('steelblue')
rects[3].set_color('goldenrod')
rects[4].set_color('darkmagenta')

rects[0].set_hatch('+')
rects[1].set_hatch('/')
rects[2].set_hatch('.')
rects[3].set_hatch('|')
rects[4].set_hatch('*')
plt.legend(loc=1)
fig.savefig('D:\SVM\plot.png')
plt.show()
       
