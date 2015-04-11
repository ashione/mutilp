"""
This example demonstrates the "ggplot" style, which adjusts the style to
emulate ggplot_ (a popular plotting package for R_).

These settings were shamelessly stolen from [1]_ (with permission).

.. [1] http://www.huyng.com/posts/sane-color-scheme-for-matplotlib/

.. _ggplot: http://had.co.nz/ggplot/
.. _R: http://www.r-project.org/

"""
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')



# bar graphs

#y1, y2,y3,y4 = np.random.randint(1, 25, size=(2, 5))
sift = [0.1,0.313,0.082,0.081,0.191,0.123,0.129,0.348,0.458,0.161,0.142,0.262,0.2]
hog_hof = [0.088,0.749,0.263,0.675,0.09,0.116,0.135,0.496,0.537,0.316,0.072,0.035,0.324]
shf = [0.107,0.075,0.286,0.571,0.116,0.141,0.138,0.556,0.565,0.278,0.078,0.325,0.326]
this = [0.2778 ,0.893,0.3543,0.3616,0.432,0.3914,0.2897,0.581,0.6632,0.4671,0.0956,0.4944,0.4418]
width = 0.2
ax  = plt.subplot()

x = np.arange(len(this))

ax.bar(x, sift, width)
ax.bar(x+width, hog_hof, width, color=plt.rcParams['axes.color_cycle'][2])
ax.bar(x+2*width, shf, width, color=plt.rcParams['axes.color_cycle'][5])
ax.bar(x+3*width, this, width, color=plt.rcParams['axes.color_cycle'][3])

ax.set_xticks(x+width)

ax.set_xticklabels(['AnswerPhone', 'DriveCar', 'Eat', 'Fight', 'GetOutCar','HandShake','HugPerson','Kiss','Run','SitDown','Situp','StandUp','Map'])
ax.set_xlabel('category')
ax.set_ylabel('MAP')
ax.legend(['SIFT','HOG+HOF','SIFT+HOG+HOF','3D CNN'])
plt.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None) 
plt.xticks(rotation=40)
plt.show()
