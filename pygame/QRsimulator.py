"""
Created on Thu Mar 19 10:30:03 2020

Simulator of a quantum repeater

@author: alessandroseri
"""
import os
import numpy as np
import pygame
from random import random
import copy
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 100, 150)
BROWN = (120,60,0)
FALDA = (60,60,240)
SIGNAL = (200,200,0)
IDLER = (200,0,200)

class QRL:
    def position(self, x = 0, y=0):
        self.posx = x
        self.posy = y
    def __init_QRL__(self, tel_d=1000,eff=0.1,tel_loss=.017, rate=1):
        self.telecom_distance = tel_d
        self.memory_efficiency=mem_eff
        self.telecom_losses=tel_loss #in db/km
        self.generation_rate=rate
    def readout(self,read=True):
        self.readout=True
    def drawme(self,screen,image_path,image_width_resized=300):
        global image_width
        global image_height
        global image_center
        image_folder='QRL2'
        image_QRL = pygame.image.load(os.path.join(image_folder,image_path))
        image_QRL.convert_alpha()
        image_width=image_QRL.get_width()
        image_height=image_QRL.get_height()
        image_QRL = pygame.transform.scale(image_QRL, (int(image_width_resized), int(image_height/image_width*image_width_resized)))
        image_width=image_QRL.get_width()
        image_height=image_QRL.get_height()
        image_center = (self.posx-int(image_width/2),self.posy-int(image_height/2))
        screen.blit(image_QRL,image_center)

    def start_generation(self,image_center=(500,266),which_source=True):
        global image_width
        global image_height
        if which_source:
            i1_x0 = int(image_center[0]-.2*image_width)
            i1_y0 = int(image_center[1]+.5*image_height*.8)
            s1_x0 = int(image_center[0]-.2*image_width)
            s1_y0 = int(image_center[1]+.5*image_height*.8)
#            which_source_factor=1
#            pp1.append([i1_x0,i1_y0,s1_x0,s1_y0])
        else:
            i1_x0 = int(image_center[0]+.22*image_width)
            i1_y0 = int(image_center[1]+.5*image_height*.8)
            s1_x0 = int(image_center[0]+.22*image_width)
            s1_y0 = int(image_center[1]+.5*image_height*.8)
#            which_source_factor=-1
#            pp2.append([i1_x0,i1_y0,s1_x0,s1_y0])
        pp = np.array([i1_x0,i1_y0,s1_x0,s1_y0])
    #    started=True
        return pp
    
def movement(pp_all, pp_starting, s_step,s_movx,s_movy, i_step,i_movx,i_movy, step_done,QR_number):
    pp_all=copy.deepcopy(pp_starting)
    idler_arrived = np.zeros((QR_number,2))
    for jj in range(int(len(pp_starting)/2)):
        sign=1
        if pp_starting[2*jj+1][1]:
                sign=sign*(-1)
        if (s_step>=step_done[jj][0]):
            pp_all[2*jj][2]=pp_starting[2*jj][2] - s_movx*sign*step_done[jj][0]
            pp_all[2*jj][3]=pp_starting[2*jj][3] - s_movy*step_done[jj][0]
        else:
            pp_all[2*jj][2]=pp_starting[2*jj][2] - s_movx*sign*s_step
            pp_all[2*jj][3]=pp_starting[2*jj][3] - s_movy*s_step
            
        if not(pp_starting[2*jj+1][1]):
                sign=sign*1.
        if (i_step>=step_done[jj][0]):
            pp_all[2*jj][0]=pp_starting[2*jj][0] + i_movx*sign*step_done[jj][0]
            pp_all[2*jj][1]=pp_starting[2*jj][1] - i_movy*step_done[jj][0]
        else:
            pp_all[2*jj][0]=pp_starting[2*jj][0] + i_movx*sign*i_step
            pp_all[2*jj][1]=pp_starting[2*jj][1] - i_movy*i_step
            if pp_starting[2*jj+1][1]:
                idler_arrived[pp_starting[2*jj+1][0]][0]+=1
            else:
                idler_arrived[pp_starting[2*jj+1][0]][1]+=1
        step_done[jj][0]+=1
#    step_done = [step_done]
    return pp_all,step_done,idler_arrived

def fiber_loss(pp_all,pp_starting,step_done, tel_distance, i_step, tel_loss,i_lost):
    pp_lost_idler=[]
    probability_transmission = 10**(-tel_loss/10*(tel_distance*1e-3/i_step))
    bb=0
    while bb in range(int(len(pp_all)/2)):
        if (random()>probability_transmission):
            pp_lost_idler.append(pp_all[2*bb:2*bb+2])       
            del pp_all[2*bb:2*bb+2]
            del pp_starting[2*bb:2*bb+2]
#            pp_all.remove(pp_all[2*ss])
            step_done.remove(step_done[bb])
            i_lost +=1
            print('lost')
            bb-=1
        bb+=1
    return pp_all,pp_starting,step_done,i_lost,pp_lost_idler

def memory_loss(pp_all,pp_starting,step_done,storage_time,time_passed,s_lost):
    bb=0
    while bb in range(int(len(pp_all)/2)):
        if (time_passed-pp_all[2*bb+1][2]>storage_time):
            del pp_all[2*bb:2*bb+2]
            del pp_starting[2*bb:2*bb+2]
#            pp_all.remove(pp_all[2*ss])
            step_done.remove(step_done[bb])
            s_lost +=1
            print('lost')
            bb-=1
        bb+=1
    return pp_all,pp_starting,step_done,s_lost#,pp_lost_s

def check_idler_arrived_undist(idler_arrived,idler_arrived_undist):
    idler_arrived_undist = np.zeros(QR_number+1)
    for aa in range(len(idler_arrived)):
        for bb in range(len(idler_arrived[0])):
            idler_arrived_undist[aa+bb]+=idler_arrived[aa,bb]
    return idler_arrived_undist
    
pygame.init()
 
# Set the width and height of the screen [width, height]
screen_width = 1200
screen_height = 400
pp_radius = int(screen_height/100)
size = (screen_width, screen_height)
screen = pygame.display.set_mode(size)

#movement = 5
pygame.display.set_caption("Quantum Repeater Simulator")
 

#Parameters
clock_ticking = 2
t_1sec_resolution=10e-6 #how much does 1 s mean in the simulation
rate = 10000 # generation per second
prob_generation = t_1sec_resolution/clock_ticking*rate
# put the p 

tel_distance=10e3
mem_eff=.1
tel_loss=.17
storage_time=50e-6

QR_number = 3
image_width_rescale = screen_width/(QR_number+3/4)
s_step = 2
i_step = int(tel_distance/(3e8)/(t_1sec_resolution)*clock_ticking)
#global image_width, image_height

graphic_distance = 14.61385/27.77*image_width_rescale
s_movx = image_width_rescale*(2.75/28.42)/(s_step)
s_movy = image_width_rescale/28.42*13.6*(3/13.6)/(s_step)
i_movx = graphic_distance*np.sin(np.arctan(8.18/12.16))/(i_step)
i_movy = graphic_distance*np.cos(np.arctan(8.18/12.16))/(i_step)

QR_posy = int(2/3*screen_height)
QR_posx = int(screen_width/(QR_number+1))
myQR=[];
myQR_center=np.zeros((QR_number,2))
myQR_width=np.zeros(QR_number)
myQR_heigth=np.zeros(QR_number)
for ii in range(QR_number):
    myQR.append(QRL())
    myQR[ii].__init_QRL__(tel_distance,mem_eff, tel_loss,rate)
    myQR[ii].position(QR_posx*(ii+1),QR_posy)


# Loop until the user clicks the close button.
done = False
pp_generated=0
pp_all=[]
pp_starting=[]
step_done=[]
idler_arrived_bool = [False for cc in range(QR_number*2)] 
i_lost=0
pp_lost_idler=[]
s_lost=0
generated = np.zeros((QR_number,2))
idler_arrived = np.zeros((QR_number,2))
idler_arrived_undist = np.zeros(QR_number+1)

time_passed=0
# Used to manage how fast the screen updates
clock = pygame.time.Clock()
 
# -------- Main Program Loop -----------
while not done:
    # --- Main event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
 
    # --- Game logic should go here
 
    # --- Screen-clearing code goes here
 
    # Here, we clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.
 
    # If you want a background image, replace this clear with blit'ing the
    # background image.
    screen.fill(WHITE)
    # --- Drawing code should go here
    idler_arrived_undist = check_idler_arrived_undist(idler_arrived,idler_arrived_undist)
    
    
    for ii in range(QR_number):
        image_path='1.png'
        if idler_arrived_undist[ii]>0:
            image_path='2l.png'
        if idler_arrived_undist[ii+1]>0:
            image_path='2r.png'
        if (idler_arrived_undist[ii]>0 and idler_arrived_undist[ii+1]>0):
            image_path='2lr.png'

        myQR[ii].drawme(screen,image_path,image_width_rescale)
        global image_center
        global image_width, image_height
#        myQR_width[ii]=image_width
#        myQR_heigth[ii]=image_height
        myQR_center[ii,0]=image_center[0]+int(image_width/2)
        myQR_center[ii,1]=image_center[1]+int(image_height/2)
        start_gen=random()<prob_generation
        whichsource = random()<.5
        if start_gen:
            pp = myQR[ii].start_generation(myQR_center[ii],whichsource)
            pp_starting.append(pp)
            pp_starting.append([ii,whichsource,time_passed])
            pp_generated+=1
            step_done.append([0])
            if whichsource:
                generated[ii,0]+=1
                myQR[ii].drawme(screen,'1l.png',(screen_width/(QR_number+3/4)))
            else:
                generated[ii,1]+=1
                myQR[ii].drawme(screen,'1r.png',(screen_width/(QR_number+3/4)))
    pp_all,pp_starting, step_done, i_lost,pp_lost_idler = fiber_loss(pp_all,pp_starting,step_done, tel_distance, i_step, tel_loss,i_lost)
    pp_all,pp_starting, step_done, s_lost = memory_loss(pp_all,pp_starting,step_done,storage_time,time_passed,s_lost)
    for aa in range(int(len(pp_all)/2)):
        pygame.draw.circle(screen,IDLER,pp_all[2*aa][0:2],pp_radius)
        pygame.draw.circle(screen,SIGNAL,pp_all[2*aa][2:4],int(pp_radius/3*2))
        
    for aa in range(int(len(pp_lost_idler)/2)):
        pygame.draw.circle(screen,BLACK,pp_all[2*aa][0:2],pp_radius)
        pygame.draw.circle(screen,BLACK,pp_all[2*aa][2:4],int(pp_radius/3*2))
        
#    if step_done!=[]:
    pp_all,step_done,idler_arrived=movement(pp_all, pp_starting, s_step,s_movx,s_movy, i_step,i_movx,i_movy, step_done,QR_number)
    
    # ---- Time passed
    time_passed +=t_1sec_resolution/clock_ticking
    font = pygame.font.Font('freesansbold.ttf', 20) 
    text = font.render('Time passed {0:.1f} us'.format(time_passed*1e6), True, BLACK, WHITE) 
    screen.blit(text, (10,10))
    text = font.render('Photon pair generated {0:.0f}, lost in fiber {1:.0f} and in the memroy {2:.0f}'.format(np.sum(generated),i_lost,s_lost), True, BLACK, WHITE) 
    screen.blit(text, (10,40))
    pygame.display.flip()


    # --- Limit to 60 frames per second
    
    clock.tick(clock_ticking)
 
# Close the window and quit.
pygame.quit()


#
#if event.type == pygame.MOUSEBUTTONDOWN:
#            # Set the x, y postions of the mouse click
#            x, y = event.pos
#            if redSquare.get_rect().collidepoint(x, y):
#                print('clicked on image')