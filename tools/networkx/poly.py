# -*- coding: utf-8 -*-
import glog
class StraightLineVerticalException(Exception):pass
class StraightLineHorizontalException(Exception):pass

def get_slope(p1,p2):
    if p1[0]==p2[0]:
        raise StraightLineVerticalException
    if p1[1]==p2[1]:
        raise  StraightLineHorizontalException
    return float(p2[1]-p1[1])/float(p2[0]-p1[0])

def get_straight_line(p1,p2):
    k=get_slope(p1,p2)
    return k,p1[1]-k*p1[0]

def get_intersect(y,p1,p2):
    k,b=get_straight_line(p1,p2)
    return ((y-b)/k,y)

def check_in_line(p,p1,p2):
    if p2[0]<p1[0]:
        p1,p2=p2,p1
    if p2[1]<p1[1]:
        return p[0]>p1[0] and p[0]<=p2[0] and p[1]<p1[1] and p[1]>=p2[1]
    else:
        return p[0]>p1[0] and p[0]<=p2[0] and p[1]>p1[1] and p[1]<=p2[1]

def check_intersect(p,p1,p2):
    try:
        intersect=get_intersect(p[1],p1,p2)
    except StraightLineVerticalException:
        return p[0]<=p1[0] and p[1]>p1[1] and p[1]<=p2[1]
    except StraightLineHorizontalException:
        return False
    if check_in_line(intersect,p1,p2):
        return intersect[0]>=p[0]
    return False

def check_point_in_polygon(point,polygon):
    point=(float(point[0]),float(point[1]))
    intersect=0
    for i in xrange(len(polygon)):
        if check_intersect(point,polygon[i],polygon[i+1 if i+1<len(polygon) else 0]):
            intersect+=1
    return intersect%2

def check_polygon_cross_polyon(polygon1,polygon2):
    glog.info('polygon : {0},type : {1}'.format(polygon1,type(polygon1)))
    return sum(map(lambda point : check_point_in_polygon(point,polygon2),polygon1)) == len(polygon1)
