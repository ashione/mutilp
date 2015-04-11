"""
This shows an example of the "fivethirtyeight" styling, which
tries to replicate the styles from FiveThirtyEight.com.
"""


from matplotlib import pyplot as plt
import numpy as np

#x= [ x*5000 for x in range(1,9)]
x = range(1,11)
orginal =map(lambda x: x/100,[15.20 ,17.30 ,19.99, 22.34,24.98,24.82,24.44,24.88,24.99,25.02])
farneback = map(lambda x: x/100,[20.33,24.45,26.14 ,29.68 ,31.14, 32.11 ,32.14 ,32.34 ,32.14 ,32.25 ])
epicflow = map(lambda x: x/100,[30.43 ,34.88,36.91 ,38.67 ,40.10 ,41.65 ,41.42 ,41.52 ,41.78 ,41.69 ])

orignal2= map(lambda x: x/100,[14.25 ,17.88 ,	22.76 ,	24.33 ,	24.76 ,	25.13 ,	25.90 ,	26.19 ])
farneback2 = map(lambda x: x/100,[17.66 ,23.90 ,	28.36 ,	30.76 ,	32.13 ,	32.74 ,	33.88 ,	34.17 ])
epicflow2 = map(lambda x: x/100,[19.64 ,27.98 ,	34.71 ,	39.99 ,	41.03 ,	41.50 ,	42.23 ,	42.56 ])

x2 = [ t*5 for t in range(1,9)]

ax= plt.subplot(1,2,1);
ax.plot(x, orginal,'o-',markersize=10)
ax.plot(x, farneback,'*-',markersize=10)
ax.plot(x, epicflow,'<-',markersize=10)
ax.set_autoscale_on(True)
ax.set_xlabel('Length of Frames')
ax.set_ylabel('MAP')
ax.set_title('a) Continuous Frames')
ax.legend(['Original','Farneback','EpicFlow'],loc=4)

ax2= plt.subplot(1,2,2);
ax2.plot(x2, orignal2,'o-',markersize=10)
ax2.plot(x2, farneback2,'*-',markersize=10)
ax2.plot(x2, epicflow2,'<-',markersize=10)
ax2.set_autoscale_on(True)
ax2.set_xlabel('Number of Iterations (k)')
ax2.set_ylabel('MAP')
ax2.set_title('b) Iteration')
ax2.legend(['Original','Farneback','EpicFlow'],loc=4)

#plt.xticks(rotation=8)
plt.subplots_adjust(left=0.1, bottom=0.1, right=None, top=None, wspace=0.3, hspace=0.1) 
plt.show()
plt.savefig('fig4.eps')
