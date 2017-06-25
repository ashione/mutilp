#!/usr/bin/python
import Image,ImageDraw,ImageFilter
import random
fix_bound = (512,512)
ratio = fix_bound[0]/50

def create_image(mode,size,pix):
    return Image.new(mode,size,pix)

def colorRandom_RGB():
    return (random.randint(64,255),random.randint(64,255),random.randint(64,255))

def colorRandom_L():
    return random.randint(0,127)
 
def random_position(x_bound,y_bound):
    return random.randint(x_bound[0],x_bound[1]), random.randint(y_bound[0],y_bound[1])

def rectangle_random(x_bound,y_bound):
    return random.choice(range(x_bound[0],x_bound[1]+1)), random.choice(range(y_bound[0],y_bound[1]+1))

def check_bound(x,y):
    return x>=0 and x < fix_bound[0] and y>=0 and y < fix_bound[1]

if __name__ == '__main__':
    noise_img = create_image(mode = 'L',size=fix_bound,pix=255)
    draw = ImageDraw.Draw(noise_img)
    for i in range(random.randint(5,10)):
        position = (random_position((0,fix_bound[0]),(0,fix_bound[1])))
        pix = colorRandom_L()
        #noise_bound_x = (position[0]-random.randint(ratio/2,ratio)*ratio,position[0]+random.randint(ratio/2,ratio)*ratio)
        #noise_bound_y = (position[1]-random.randint(ratio/2,ratio)*ratio,position[1]+random.randint(ratio/2,ratio)*ratio)
        noise_bound_x = (position[0]-random.randint(0,ratio),position[0]+random.randint(0,ratio))
        noise_bound_y = (position[1]-random.randint(0,ratio),position[1]+random.randint(0,ratio))
        #print noise_bound_x,noise_bound_y

        for j in range(random.randint(ratio/2,ratio)*ratio):
            position = (rectangle_random(noise_bound_x,noise_bound_y))
            pix = colorRandom_L()
            draw.point(position,fill=pix)
    noise_img.show()
