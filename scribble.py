import cv2
import numpy as np

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
current_former_x,current_former_y = -1,-1

def nothing(x):
    pass

# mouse callback function
def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode
 
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
 
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(image,(current_former_x,current_former_y),(former_x,former_y),(b,g,r),5)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(image,(current_former_x,current_former_y),(former_x,former_y),(b,g,r),5)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y
 
def main(): 
    image = cv2.imread('11.png')
    cv2.namedWindow('image')
    #cv2.namedWindow('trackbar')
    #cv2.resizeWindow('trackbar', (10,10))
    cv2.setMouseCallback('image', paint_draw)
    cv2.createTrackbar('R','image',0,255, nothing)
    cv2.createTrackbar('G','image',0,255, nothing)
    cv2.createTrackbar('B','image',0,255, nothing)
    while(1):
        cv2.imshow('image',image)
        if cv2.waitKey(20) & 0xFF == 27:
            break
        r = cv2.getTrackbarPos('R','image')
        g = cv2.getTrackbarPos('G','image')
        b = cv2.getTrackbarPos('B','image')
    cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
