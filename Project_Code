# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:41:10 2020

@author: joshu
"""

from vpython import *
import numpy as np
from random import uniform,choice
from math import sqrt
import matplotlib.pyplot as plt
import scipy.stats as stats
from copy import deepcopy



class ParticlesInBox(object):
    
   
    def __init__(self,numpar = 500,velocity=100.0,radius=5.0,mass = 1.0,time = 10,time_step = 0.005):
        self.numpar = numpar
        self.velocity = velocity / sqrt(2.0)  #sqrt(2) so i can split it up evenly to Vx and Vy   
        self.radius = radius
        self.mass = mass
        self.time = time
        self.time_step = time_step
        self.side_length = 1000.0
        
        
        
    def initparticles(self):
        #This method is for creating particles given mass,radius, and velocity
        x_position = uniform(-self.side_length,self.side_length)
        y_position = uniform(-self.side_length,self.side_length)
        random_direction_x = choice((-1,1))
        random_direction_y = choice((-1,1))
        #Creating the particle
        particle = sphere(pos=vec(x_position,y_position,0),radius = self.radius, color = vec(0,0,0), mass = self.mass)
        #Only x and y components so velocity.z = 0
        particle.velocity = vec(self.velocity * random_direction_x,self.velocity * random_direction_y,0)
        return particle
    
    
    def initparticles_twogases(self,velocity2,radius2,mass2):
        #This method is for Two types of particle. You need to feed mass, radius, and velocity of particle 2
        x_position = uniform(-self.side_length,self.side_length)
        y_position = uniform(-self.side_length,self.side_length)
        random_direction_x = choice((-1,1))
        random_direction_y = choice((-1,1))
        #Creating the particle
        particle = sphere(pos=vec(x_position,y_position,0),radius = radius2, color = vec(0.4,0.2,0.6), mass = mass2)
        #Only x and y components so velocity.z = 0
        particle.velocity = vec(velocity2 * random_direction_x / sqrt(2.0),velocity2 * random_direction_y / sqrt(2.0),0)
        return particle
    
    def particle_list(self):
        part_list = []
        #Creates a list of particles with same mass,radius, velocity
        for i in range(0,self.numpar):
            particle = self.initparticles()
            part_list.append(particle)

        return part_list
    
    def twoparticle_list(self,numpar2,velocity2,radius2,mass2):
        #Finding out how many are particle 1
        numpar1 = self.numpar - numpar2
        part_list = []
        #Loop for creating particle 1 and append to particle list
        for i in range(0,numpar1):
            particle = self.initparticles()
            part_list.append(particle)
        #Loop for creating particle 2 and append to particle list
        for j in range(0,numpar2):
            particle = self.initparticles_twogases(velocity2,radius2,mass2)
            part_list.append(particle)
    
        return part_list
        
    
    def initwall(self):
        #This method creates the box in the animation
        thickness = 10.0
        right = box(pos=vec(self.side_length, 0, 0), length=0.01, height=self.side_length, width=thickness, color = vec(1,1,1))  
        left = box(pos=vec(-self.side_length, 0, 0), length=0.01, height=self.side_length, width=thickness, color = vec(1,1,1))  
        bottom = box(pos=vec(0, -self.side_length, 0), length=self.side_length, height=0.01, width=thickness, color = vec(1,1,1))  
        top = box(pos=vec(0, self.side_length, 0), length=self.side_length, height=0.01, width=thickness, color = vec(1,1,1)) 
        back = box(pos=vec(0, 0, -self.side_length), length=2.55*self.side_length, height=2.55*self.side_length, width=thickness, color = vec(1,1,1))
        return right,left,bottom,top,back

    def onepar_simulation(self):
        #This method creates 1 particle inside a box
        particle = self.initparticles()
        wall = self.initwall()
        sim_time = 0      
        
        while sim_time < self.time:
            #Adding time to the simulation Clock
            sim_time = sim_time + self.time_step
            print (sim_time)
            #Runs visual aspect rate(n) on n per second basis
            rate(100)
            
            #Updating the particles position
            particle.pos = particle.pos + particle.velocity*self.time_step
            
            
            #Small displacement
            delta_distance_x = particle.velocity.x*self.time_step
            delta_distance_y = particle.velocity.y*self.time_step
            
            #When it hits the right wall
            if (particle.pos.x + self.radius) > self.side_length:
                #Changes the velocity to the other direction
                particle.velocity.x = -particle.velocity.x
                
                #Changes the position of the particle by a small displacement
                particle.pos.x = self.side_length - self.radius - delta_distance_x
                
            #When it hits the left wall
            elif (particle.pos.x - self.radius) < -self.side_length:
                particle.velocity.x = -particle.velocity.x
                particle.pos.x = -(self.side_length - self.radius) + delta_distance_x
                
            #When it hits the top wall
            if (particle.pos.y + self.radius) > self.side_length:
                particle.velocity.y = -particle.velocity.y
                particle.pos.y = (self.side_length - self.radius) - delta_distance_y
               
            #When it hits the bottom wall
            elif (particle.pos.y - self.radius) < -self.side_length:
                particle.velocity.y = -particle.velocity.y
                particle.pos.y = - (self.side_length - self.radius) + delta_distance_y
            

    def multipar_simulation(self):
        #This method creates multiple particles inside a box
        part_list = self.particle_list()
        wall = self.initwall()
        sim_time = 0 
        #Had to use the package deepcopy because initial_conditions = part_list wouldn't work because when part_list updates initial_conditions also changes.
        initial_conditions = deepcopy(part_list)
        while sim_time < self.time:
            sim_time = sim_time + self.time_step
            print(sim_time)
            #Runs visual aspect rate(n) on n per second basis
            rate(1000)
            
            #For loop for updating each particle position and detecting wall collision
            for particle in part_list:

                #Updating the particles position
                particle.pos = particle.pos + particle.velocity*self.time_step
                
                #Adding time to the simulation Clock
                #sim_time = sim_time + self.time_step
                
                #Small displacement
                delta_distance_x = particle.velocity.x*self.time_step
                delta_distance_y = particle.velocity.y*self.time_step
                
                #When it hits the right wall
                if (particle.pos.x + self.radius) > self.side_length:
                    #Changes the velocity to the other direction
                    particle.velocity.x = -particle.velocity.x
                    
                    #Changes the position of the particle by a small displacement
                    particle.pos.x = self.side_length - self.radius - delta_distance_x
                    
                #When it hits the left wall
                elif (particle.pos.x - self.radius) < -self.side_length:
                    particle.velocity.x = -particle.velocity.x
                    particle.pos.x = -(self.side_length - self.radius) + delta_distance_x
                    
                #When it hits the top wall
                if (particle.pos.y + self.radius) > self.side_length:
                    particle.velocity.y = -particle.velocity.y
                    particle.pos.y = (self.side_length - self.radius) - delta_distance_y
                   
                #When it hits the bottom wall
                elif (particle.pos.y - self.radius) < -self.side_length:
                    particle.velocity.y = -particle.velocity.y
                    particle.pos.y = - (self.side_length - self.radius) + delta_distance_y
            
            #For Loop for particle collision
            for tar_part_index in range(0,self.numpar):
                for other_part_index in range(tar_part_index + 1,self.numpar):
                    #Calculates the distance between the two particles from their center
                    abs_distance = sqrt((part_list[tar_part_index].pos.x - part_list[other_part_index].pos.x)**2 + (part_list[tar_part_index].pos.y - part_list[other_part_index].pos.y)**2)
                    #if statement for changing the momentum
                    if abs_distance < (part_list[tar_part_index].radius + part_list[other_part_index].radius):
                        #Renaming the particles
                        part1 = part_list[tar_part_index]
                        part2 = part_list[other_part_index]
                        
                        #Final Velocity of Particle 1. Formula taken from "Elastic Collision" Wikipedia
                        coeffcient1 = (2*part2.mass)/(part1.mass + part2.mass)
                        inner_prod1 = dot(part1.velocity - part2.velocity,part1.pos - part2.pos) 
                        magnitude1 = mag2(part1.pos - part2.pos)
                        direction1 = (part1.pos - part2.pos)
                        v1_final = part1.velocity - coeffcient1*(inner_prod1/magnitude1)*direction1
                        
                        #Final Velocity of Particle 2. Formula taken from "Elastic Collision" Wikipedia
                        coeffcient2 = (2*part1.mass)/(part2.mass + part2.mass)
                        inner_prod2 = dot(part2.velocity - part1.velocity,part2.pos - part1.pos) 
                        magnitude2 = mag2(part2.pos - part1.pos)
                        direction2 = (part2.pos - part1.pos)
                        v2_final = part2.velocity - coeffcient2*(inner_prod2/magnitude2)*direction2
                        
                        #Update Velocities
                        part_list[tar_part_index].velocity = v1_final
                        part_list[other_part_index].velocity = v2_final
            
        return (part_list,initial_conditions)
    
    
        
    def MovingWall_simulation(self,wallspeed = 150):
        #This method is for running the moving wall simulation
        part_list = self.particle_list()
        wall = self.initwall()
        sim_time = 0
        #This list keeps track of v^2 of the particles. Mainly used for analysis later
        velocitysquare_list = np.zeros(self.numpar)
        #This list keeps track of the volume each time the wall moves.
        volume_list = []
        #This list keeps track of the pressure each time the wall moves.
        pressure_list = []
        while sim_time < self.time:
            sim_time = sim_time + self.time_step
            #This gives the new location of the wall after it moves in each timestep
            movingwall_location = self.side_length - wallspeed*sim_time
            #This draws a wall in the animation
            left = box(pos=vec(-movingwall_location, 0, 0), length=0.01, height=2.55*self.side_length, width=1.0, color = vec(1,1,1))  
            print (sim_time,movingwall_location)
            #Runs visual aspect rate(n) on n per second basis
            rate(100)
            #For loop for updating each particle position and detecting wall collision
            for particle in part_list:

                #Updating the particles position
                particle.pos = particle.pos + particle.velocity*self.time_step
                
                #Adding time to the simulation Clock
                #sim_time = sim_time + self.time_step
                
                #Small displacement
                delta_distance_x = particle.velocity.x*self.time_step
                delta_distance_y = particle.velocity.y*self.time_step
                
                #When it hits the right wall
                if (particle.pos.x + self.radius) > self.side_length:
                    #Changes the velocity to the other direction
                    particle.velocity.x = -particle.velocity.x
                    
                    #Changes the position of the particle by a small displacement
                    particle.pos.x = self.side_length - self.radius - delta_distance_x
                    
                #When it hits the left wall. Since the left wall is the one moving this condition changes every time step
                elif (particle.pos.x - self.radius) < -movingwall_location:
                    particle.velocity.x = -particle.velocity.x
                    particle.pos.x = -(movingwall_location - self.radius) + delta_distance_x
                    
                #When it hits the top wall
                if (particle.pos.y + self.radius) > self.side_length:
                    particle.velocity.y = -particle.velocity.y
                    particle.pos.y = (self.side_length - self.radius) - delta_distance_y
                   
                #When it hits the bottom wall
                elif (particle.pos.y - self.radius) < -self.side_length:
                    particle.velocity.y = -particle.velocity.y
                    particle.pos.y = - (self.side_length - self.radius) + delta_distance_y
            
            #For Loop for particle collision
            for tar_part_index in range(0,self.numpar):
                for other_part_index in range(tar_part_index + 1,self.numpar):
                    #Calculates the distance between the two particles from their center
                    abs_distance = sqrt((part_list[tar_part_index].pos.x - part_list[other_part_index].pos.x)**2 + (part_list[tar_part_index].pos.y - part_list[other_part_index].pos.y)**2)
                    #if statement for changing the momentum
                    if abs_distance < (part_list[tar_part_index].radius + part_list[other_part_index].radius):
                        #Renaming the particles
                        part1 = part_list[tar_part_index]
                        part2 = part_list[other_part_index]
                        
                        #Final Velocity of Particle 1. Formula taken from "Elastic Collision" Wikipedia
                        coeffcient1 = (2*part2.mass)/(part1.mass + part2.mass)
                        inner_prod1 = dot(part1.velocity - part2.velocity,part1.pos - part2.pos) 
                        magnitude1 = mag2(part1.pos - part2.pos)
                        direction1 = (part1.pos - part2.pos)
                        v1_final = part1.velocity - coeffcient1*(inner_prod1/magnitude1)*direction1
                        
                        #Final Velocity of Particle. Formula taken from "Elastic Collision" Wikipedia
                        coeffcient2 = (2*part1.mass)/(part2.mass + part2.mass)
                        inner_prod2 = dot(part2.velocity - part1.velocity,part2.pos - part1.pos) 
                        magnitude2 = mag2(part2.pos - part1.pos)
                        direction2 = (part2.pos - part1.pos)
                        v2_final = part2.velocity - coeffcient2*(inner_prod2/magnitude2)*direction2
                        
                        #Update Velocities
                        part_list[tar_part_index].velocity = v1_final
                        part_list[other_part_index].velocity = v2_final
        
                #Finding the square velocity of the particle for calculating pressure later and appending to list
                velocity_square = mag2(part_list[tar_part_index].velocity)
                velocitysquare_list[tar_part_index] = velocity_square
            
            #Calculating volume in that time-step
            volume = (2*self.side_length - wallspeed*sim_time)*(2*self.side_length)
            
            #First calculate the mean square volume and using formula to find pressure
            velocitysquare_list = np.array(velocitysquare_list)
            meansquare_velocity = np.mean(velocitysquare_list)
            pressure = ((self.numpar*self.mass)/(3*volume))*meansquare_velocity
            
            #Appending values to respective list
            volume_list.append(volume)
            pressure_list.append(pressure)
            
        return (part_list,volume_list,pressure_list)


    def twogases_simulation(self,input_numpar2,input_velocity2,input_radius2,input_mass2):
        #Needed to use the method for creating a list two different particles
        part_list = self.twoparticle_list(input_numpar2,input_velocity2,input_radius2,input_mass2)
        #Had to use the package deepcopy because initial_conditions = part_list wouldn't work because when part_list updates initial_conditions also changes.
        initial_conditions = deepcopy(part_list)
        wall = self.initwall()
        sim_time = 0
        while sim_time < self.time:
            print (sim_time)
            sim_time = sim_time + self.time_step
            #Runs visual aspect rate(n) on n per second basis
            rate(1000)
            #For loop for updating each particle position and detecting wall collision
            for particle in part_list:

                #Updating the particles position
                particle.pos = particle.pos + particle.velocity*self.time_step
                
                #Small displacement
                delta_distance_x = particle.velocity.x*self.time_step
                delta_distance_y = particle.velocity.y*self.time_step
                
                #When it hits the right wall
                if (particle.pos.x + self.radius) > self.side_length:
                    #Changes the velocity to the other direction
                    particle.velocity.x = -particle.velocity.x
                    
                    #Changes the position of the particle by a small displacement
                    particle.pos.x = self.side_length - self.radius - delta_distance_x
                    
                #When it hits the left wall
                elif (particle.pos.x - self.radius) < -self.side_length:
                    particle.velocity.x = -particle.velocity.x
                    particle.pos.x = -(self.side_length - self.radius) + delta_distance_x
                    
                #When it hits the top wall
                if (particle.pos.y + self.radius) > self.side_length:
                    particle.velocity.y = -particle.velocity.y
                    particle.pos.y = (self.side_length - self.radius) - delta_distance_y
                   
                #When it hits the bottom wall
                elif (particle.pos.y - self.radius) < -self.side_length:
                    particle.velocity.y = -particle.velocity.y
                    particle.pos.y = - (self.side_length - self.radius) + delta_distance_y
            
            #For Loop for particle collision
            for tar_part_index in range(0,self.numpar):
                for other_part_index in range(tar_part_index + 1,self.numpar):
                    #Calculates the distance between the two particles from their center
                    abs_distance = sqrt((part_list[tar_part_index].pos.x - part_list[other_part_index].pos.x)**2 + (part_list[tar_part_index].pos.y - part_list[other_part_index].pos.y)**2)
                    #if statement for changing the momentum
                    if abs_distance < (part_list[tar_part_index].radius + part_list[other_part_index].radius):
                        #Renaming the particles
                        part1 = part_list[tar_part_index]
                        part2 = part_list[other_part_index]
                        
                        #Final Velocity of Particle 1. Formula taken from "Elastic Collision" Wikipedia
                        coeffcient1 = (2*part2.mass)/(part1.mass + part2.mass)
                        inner_prod1 = dot(part1.velocity - part2.velocity,part1.pos - part2.pos) 
                        magnitude1 = mag2(part1.pos - part2.pos)
                        direction1 = (part1.pos - part2.pos)
                        v1_final = part1.velocity - coeffcient1*(inner_prod1/magnitude1)*direction1
                        
                        #Final Velocity of Particle. Formula taken from "Elastic Collision" Wikipedia
                        coeffcient2 = (2*part1.mass)/(part2.mass + part2.mass)
                        inner_prod2 = dot(part2.velocity - part1.velocity,part2.pos - part1.pos) 
                        magnitude2 = mag2(part2.pos - part1.pos)
                        direction2 = (part2.pos - part1.pos)
                        v2_final = part2.velocity - coeffcient2*(inner_prod2/magnitude2)*direction2
                        
                        #Update Velocities
                        
                        part_list[tar_part_index].velocity = v1_final
                        part_list[other_part_index].velocity = v2_final
                        if tar_part_index == 1:
                            print ("Collision" + str(v1_final))
        return (part_list,initial_conditions)







    
        
#particle_list,initial_conditions = ParticlesInBox(numpar = 500,velocity=300.0,radius = 7.0, mass = 2.0,time = 5, time_step = 0.005).multipar_simulation()
particle_list, vol_list, press_list = ParticlesInBox(numpar = 400,velocity=400.0,radius = 3.0, mass = 2.0,time = 18, time_step = 0.003).MovingWall_simulation(wallspeed = 100)
#particle_list = ParticlesInBox(numpar = 100,velocity=300.0,radius = 15.0, mass = 10.0,time = 2, time_step = 0.005).twogases_simulation(60,300.0, 6.0 ,6.0)


#Depending on which simulation you want to run just delete the top ''' and bottom ''' to uncomment for the plots

#Momentum test


velocityI_list = []
velocityF_list = []
for i in range(len(particle_list)):
    init_vel = mag2(initial_conditions[i].velocity)*(initial_conditions[i].mass)
    final_vel = mag2(particle_list[i].velocity)*(initial_conditions[i].mass)
    velocityI_list.append(init_vel)
    velocityF_list.append(final_vel)

velocityI_list = np.array(velocityI_list)
velocityF_list = np.array(velocityF_list)
Initial_momentum = np.sum(velocityI_list)
Final_momentum = np.sum(velocityF_list)

print ("Initial Momentum is " + str(sqrt(Initial_momentum)))
print ("Final Momentum is " + str(sqrt(Final_momentum)))


#Energy Test

initial_KE_list = []
final_KE_list = []
for i in range(len(particle_list)):
    initial_KE = 0.5*mag2(initial_conditions[i].velocity)*(initial_conditions[i].mass)
    final_KE = 0.5*mag2(particle_list[i].velocity)*(particle_list[i].mass)
    initial_KE_list.append(initial_KE)
    final_KE_list.append(final_KE)
    
initial_KE_list = np.array(initial_KE_list)
final_KE_list = np.array(final_KE_list)

print ("The Initial Kinetic Energy is " + str(np.sum(initial_KE_list)))
print ("The Final Kinetic Energy is " + str(np.sum(final_KE_list)))


'''

#For Regular Wall Simulation


velocity_list = []
rms_velocity_list = []

for i in range(len(particle_list)):
    velocity = mag(particle_list[i].velocity)
    rms_velocity = mag2(particle_list[i].velocity)
    velocity_list.append(velocity)
    rms_velocity_list.append(rms_velocity)


rms_velocity_list = np.array(rms_velocity_list)
KE_final = np.sum(rms_velocity_list)
meansquarevelocity = np.mean(rms_velocity_list)


maxwell_fit_x = np.linspace(0, 700, 500)
params = stats.maxwell.fit(velocity_list, floc=0)

print (params)
print (meansquarevelocity)
print (KE_final)


maxwell_fit_y = stats.maxwell.pdf(maxwell_fit_x, *params)
plt.hist(velocity_list, bins = 38,density=True)
plt.plot(maxwell_fit_x, maxwell_fit_y, lw=3)
plt.show()

np.save("RegularWall_Trial8_velocity", velocity_list)

'''


'''

#For Two Gases Simulation




velocity_list = []
rms_velocity_list = []
for i in range(len(particle_list)):
    velocity = mag(particle_list[i].velocity)
    rms_velocity = mag2(particle_list[i].velocity)
    velocity_list.append(velocity)
    rms_velocity_list.append(rms_velocity)


rms_velocity_list = np.array(rms_velocity_list)
KE_final = np.sum(rms_velocity_list)
meansquarevelocity = np.mean(rms_velocity_list)

maxwell_fit_x = np.linspace(0, 700, 500)
params = stats.maxwell.fit(velocity_list, floc=0)

print (params)
print (meansquarevelocity)
print (KE_final)



maxwell_fit_y = stats.maxwell.pdf(maxwell_fit_x, *params)
plt.hist(velocity_list, bins = 38,density=True)
plt.plot(maxwell_fit_x, maxwell_fit_y, lw=3)
plt.show()

particle_list = np.array(particle_list)

np.save("TwoGases_Trial5_particle", particle_list)
np.save("TwoGases_Trial5_velocity", velocity_list)

'''




'''

#For Moving Wall Simulation

velocity_list = []
rms_velocity_list = []
for i in range(len(particle_list)):
    velocity = mag(particle_list[i].velocity)
    rms_velocity = mag2(particle_list[i].velocity)
    velocity_list.append(velocity)
    rms_velocity_list.append(rms_velocity)


rms_velocity_list = np.array(rms_velocity_list)
KE_final = np.sum(rms_velocity_list)
meansquarevelocity = np.mean(rms_velocity_list)

maxwell_fit_x = np.linspace(0, 700, 500)
params = stats.maxwell.fit(velocity_list, floc=0)

print (params)
print (meansquarevelocity)
print (KE_final)


maxwell_fit_y = stats.maxwell.pdf(maxwell_fit_x, *params)
plt.hist(velocity_list, bins = 38,density=True)
plt.plot(maxwell_fit_x, maxwell_fit_y, lw=3)
plt.show()



plt.plot(vol_list,press_list)
plt.title("PV Diagram")
plt.xlabel("Volume")
plt.ylabel("Pressure")
plt.xlim(0,1.5*max(vol_list))
plt.ylim(0,1.5*max(press_list))
plt.grid()   
plt.show()


velocity_list = np.array(velocity_list)
vol_list = np.array(vol_list)
press_list = np.array(press_list)

np.save("MovingWall_Trial3_velocity", velocity_list)
np.save("MovingWall_Trial3_volume", vol_list)
np.save("MovingWall_Trial3_pressure", press_list)

'''
        
