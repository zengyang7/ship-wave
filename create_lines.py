# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:18:34 2017

@author: adamwise
"""
import numpy as np
from numpy import random

def add_line(image, image2, thickness, theta, t):
    '''
    adds a line to an image
    INPUTS:
        image           - numpy 2D array of satellite image with capillary waves
        image2          - numpy 2D array of satellite image without capillary waves
        thickness       - thickness of wake, passed in as a number of pixels
        theta           - angle between x-axis and perpendicular distance from random line to origin
        t               - perpendicular distance between center of image and random line
    OUTPUTS:
        image_w_line    - numpy 2D array with values added for the random line
        theta           - angle between x-axis and perpendicular distance from random line to origin after accounting for discrete pixel space
        t               - perpendicular distance between center of image and random line after accounting for discrete pixel space
        flag            - flag is raised if the line is in the bounds of the image
    '''
    theta = theta*np.pi/180                 # convert degree input into radians
    (Npx, Npy) = image.shape
    x = Npx - 1
    y = Npy - 1
    trig = 0                                # flag/trigger if line is out of image (happens due to t_max). 
    point_num = 0                           # keeps track of end points
    
    # convert t and theta to a single point
    pointx = t*np.cos(theta)
    pointy = t*np.sin(theta)
    m = np.tan(theta-np.pi/2)
    b = m*-pointx + pointy
    
    # coordinates for wake accounting for the thickness
    pointx1 = (t + int(thickness/2))*np.cos(theta)  # upper x-coordinate bound of wake
    pointx2 = (t - int(thickness/2))*np.cos(theta)  # lower x-coordinate bound of wake
    pointy1 = (t + int(thickness/2))*np.sin(theta)  # upper y-coordinate bound of wake
    pointy2 = (t - int(thickness/2))*np.sin(theta)  # lower y-coordinate bound of wake
    b1_o = m*-pointx1 + pointy1
    b1 = m*(-x/2) + b1_o + y/2  # shifts the y intercept from an origin at the center of the image to the origin at the bottom left
    b2_o = m*-pointx2 + pointy2
    b2 = m*(-x/2) + b2_o + y/2  # shifts the y intercept from an origin at the center of the image to the origin at the bottom left
    
    # initialize endpoint arrays
    endpointx = np.zeros(2)
    endpointy = np.zeros(2)

    # determine where the line intersects the boundaries
    if theta == np.pi/2 or theta == 3*np.pi/2:
        endpointx[0] = -y/2
        endpointx[1] = y/2
        endpointy[:] = t         
        if t < y:
            trig = 1
    else:
        if theta == np.pi:
            endpointx[:] = t
            endpointy[0] = -x/2
            endpointy[1] = x/2
            if t < x/2:
                trig = 1
        else:
            if (m*x/2 + b  < y/2) and (m*x/2 + b  > -y/2):       # right wall
                endpointx[point_num] = x/2
                endpointy[point_num] = m*x/2 + b                     
                point_num += 1
                trig = 1
                
            if (m*-x/2 + b  < y/2) and (m*-x/2 + b  > -y/2):     # left wall
                endpointx[point_num] = -x/2
                endpointy[point_num] = m*-x/2 + b
                point_num += 1
                trig = 1
            
            if ((y/2 - b)/m  < x/2) and ((y/2 - b)/m  > -x/2):     # top wall
                endpointx[point_num] = (y/2 - b)/m
                endpointy[point_num] = y/2
                point_num += 1
                trig = 1
            
            if ((-y/2 - b)/m  < x/2) and ((-y/2 - b)/m  > -x/2):     # bottom wall
                endpointx[point_num] = (-y/2 - b)/m
                endpointy[point_num] = -y/2
                trig = 1
        
    # shift coordinates from center of image to bottom-left corner
    endpointx[:] = endpointx[:] + x/2
    endpointy[:] = endpointy[:] + y/2
  
    # find the new t and theta based on integer end points
    if trig > 0:
        if (int(endpointx[0]) - int(endpointx[1])) != 0:        # checks if there's a vertical line
            m = (int(endpointy[0]) - int(endpointy[1]))/(int(endpointx[0]) - int(endpointx[1]))         # slope based on pixel coords
            b = m*(-(int(endpointx[0]) - x/2)) + int(endpointy[0]) - y/2                                # y intercept based on pixel coords
            m_deg = np.arctan2(m,1)*180/np.pi    
            if theta > np.pi:
                theta_new = 270 + m_deg
            else:
                theta_new = 90 + m_deg 
            m_t = np.tan(theta_new*np.pi/180)
            # intersection of actual line and perpendicular line going through the center
            t_x = b/(m_t - m)
            t_y = m_t*t_x
            t = np.sqrt(t_x**2 + t_y**2)
        else:
            theta_new = 180
            t = endpointx[0] - x/2
    else:
        theta_new = theta #TODO: NOT SURE THIS IS RIGHT
    
    # sort y intercepts of wake
    b = [b1, b2]
    b.sort()
    image_w_line = np.zeros([Npx, Npy]) 
    
    # adding the wake
    if theta == np.pi:                  # vertical line
        for i in range(Npx):
            for j in range(Npy):
                if  j < pointx1 + x/2 or j > pointx2 + x/2:
                    image_w_line[i,j] = image[i,j]
                else:
                    image_w_line[i,j] = image2[i,j]    
    else:
        for i in range(Npx):
            for j in range(Npy):
                if (y-i) > m*(j) + b[1] or (y-i) < m*(j) + b[0]:
                    image_w_line[i,j] = image[i,j]
                else:
                    image_w_line[i,j] = image2[i,j]
#        print("    theta: " + str(theta))
#        print("    trig: " + str(trig))
#        print("    theta_new: " + str(theta_new))
        theta = theta_new
        t = t
    
    flag = trig  # the trigger should be greater than one if the center of the wake os on the image
         
    return image_w_line, theta, t, flag


def add_lines(image, image2, thickness_p, Nl_max):
    '''
    adds an arbitrary number of lines to an image
    INPUTS:
        image           - numpy 2D array of an image to add lines to
        image2          - numpy 2D array of an image that is the data in the lines
        thickness_p     - thickness of wake, passed in as a number of pixels
        Nl_max          - maximum number of lines/wakes to add to an image
    OUTPUTS:
        image_w_lines   - numpy 2D array of an image with lines/wales
        coord           - (theta,t) for each line
    '''
    (Npx, Npy) = image.shape
    image_w_lines = image
    t_max = np.sqrt((Npy/2)**2 + (Npx/2)**2)
    theta_step = 5
    t_step = 10
    
    Nl = int(random.random()*Nl_max + 1)        # random number of lines (at least 1)
    
    coord = np.zeros([Nl,2])
    for ii in range(Nl):
        flag = 0
        while flag < 1:
            theta = random.randint(360/theta_step)*theta_step           # between 0-360 in intervals of 5 degrees
            t = random.randint((Npx/2)/(t_step)+1)*t_step                 # between 0-50 in intervals of 10 units
            image_w_lines, theta_b, t_b, flag = add_line(image_w_lines, image2, thickness_p, theta, t)
        # output for coord is (theta, t)
        coord[ii,0] = theta
        coord[ii,1] = t
    return image_w_lines, coord