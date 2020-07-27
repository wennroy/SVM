# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:22:59 2020

@author: lengwaifang
"""


import numpy as np
from matplotlib import pyplot as plt
import csv
# X = []
# target = []
# for n in range(100):
#     x = np.random.rand()
#     y = np.random.rand()
    
#     if x + y +0.2*(np.random.rand()*2-1)>1:
#         c = 1
#     else:
#         c = -1
#     X.append([x,y])
#     target.append(c)
# X = np.array(X)
# plt.scatter(X[:,0],X[:,1],c = target)
# plt.show()

def datasetobtain():
    X = np.array([[0.83284403, 0.6017228 ],
           [0.84334127, 0.66265874],
           [0.44961989, 0.93477493],
           [0.98862644, 0.2388741 ],
           [0.77690117, 0.15654921],
           [0.92952698, 0.09318927],
           [0.15124353, 0.74951663],
           [0.08446172, 0.70324693],
           [0.81387991, 0.08180302],
           [0.38916642, 0.15084904],
           [0.13896299, 0.18746948],
           [0.30368258, 0.3091562 ],
           [0.01948952, 0.79651956],
           [0.19854858, 0.81116945],
           [0.5621246 , 0.37899634],
           [0.28974754, 0.68714193],
           [0.28906121, 0.2804657 ],
           [0.48325247, 0.21347043],
           [0.17313184, 0.21961815],
           [0.36947675, 0.27598001],
           [0.43570825, 0.36009917],
           [0.71619399, 0.82483565],
           [0.0616818 , 0.1401701 ],
           [0.42776109, 0.00391448],
           [0.59020654, 0.84481593],
           [0.43286026, 0.79367317],
           [0.22847481, 0.85834098],
           [0.360699  , 0.96833697],
           [0.42328513, 0.07806614],
           [0.5706067 , 0.72143957],
           [0.1763294 , 0.20437096],
           [0.18652152, 0.02369894],
           [0.93517753, 0.75919652],
           [0.61968723, 0.19487916],
           [0.84973995, 0.00122675],
           [0.26563072, 0.75547791],
           [0.44798614, 0.53445799],
           [0.73962654, 0.47744776],
           [0.53196879, 0.85477774],
           [0.10468061, 0.79514186],
           [0.48004091, 0.01591933],
           [0.73082169, 0.50703759],
           [0.81372807, 0.34421345],
           [0.44039244, 0.17775172],
           [0.87789979, 0.36912162],
           [0.77058815, 0.83482248],
           [0.13617229, 0.29838567],
           [0.65542898, 0.99433171],
           [0.92213372, 0.23434916],
           [0.89103743, 0.58566193],
           [0.24239424, 0.16027142],
           [0.74922429, 0.72454174],
           [0.28949138, 0.11618321],
           [0.61448607, 0.9509427 ],
           [0.32661289, 0.52975997],
           [0.08527794, 0.82055268],
           [0.61500515, 0.33243054],
           [0.58276163, 0.53488587],
           [0.17489643, 0.89107309],
           [0.66582912, 0.64598907],
           [0.24120501, 0.25620227],
           [0.49025233, 0.78076393],
           [0.09743308, 0.13588332],
           [0.47884521, 0.57482371],
           [0.06470004, 0.26266302],
           [0.12876952, 0.6148045 ],
           [0.4167946 , 0.11639935],
           [0.2560839 , 0.80850641],
           [0.53412555, 0.55943639],
           [0.71248516, 0.58295954],
           [0.97209695, 0.09229902],
           [0.55282537, 0.82198241],
           [0.85828037, 0.63113791],
           [0.78144813, 0.36419898],
           [0.63227769, 0.22750856],
           [0.83097329, 0.68389471],
           [0.29694118, 0.88102278],
           [0.38162116, 0.69288284],
           [0.55950253, 0.88100675],
           [0.70574593, 0.54678492],
           [0.1897885 , 0.20303039],
           [0.97249339, 0.07106211],
           [0.48277129, 0.07127356],
           [0.78116587, 0.878858  ],
           [0.90963718, 0.16710007],
           [0.76666906, 0.88105068],
           [0.31894841, 0.28721022],
           [0.7362145 , 0.54168159],
           [0.75670746, 0.64400167],
           [0.17729008, 0.68185127],
           [0.40796135, 0.53813217],
           [0.95231093, 0.6627184 ],
           [0.40903256, 0.67650179],
           [0.14999435, 0.49647283],
           [0.15380296, 0.60935741],
           [0.06783196, 0.60216573],
           [0.10877607, 0.27265647],
           [0.47732064, 0.55490305],
           [0.92554033, 0.14448247],
           [0.67871631, 0.98600469]])
    
    y = [1,
     1,
     1,
     1,
     -1,
     1,
     -1,
     -1,
     -1,
     -1,
     -1,
     -1,
     -1,
     -1,
     -1,
     1,
     -1,
     -1,
     -1,
     -1,
     -1,
     1,
     -1,
     -1,
     1,
     1,
     1,
     1,
     -1,
     1,
     -1,
     -1,
     1,
     -1,
     -1,
     1,
     1,
     1,
     1,
     -1,
     -1,
     1,
     1,
     -1,
     1,
     1,
     -1,
     1,
     1,
     1,
     -1,
     1,
     -1,
     1,
     -1,
     -1,
     -1,
     1,
     1,
     1,
     -1,
     1,
     -1,
     1,
     -1,
     -1,
     -1,
     1,
     1,
     1,
     -1,
     1,
     1,
     1,
     -1,
     1,
     1,
     1,
     1,
     1,
     -1,
     -1,
     -1,
     1,
     1,
     1,
     -1,
     1,
     1,
     1,
     1,
     1,
     -1,
     -1,
     -1,
     -1,
     -1,
     1,
     1,
     1]
    return X,y