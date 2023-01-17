import numpy as np
import pygame
from random import randint
import math
import time
from IA import NN2
DCT = NN2()
VTS = NN2()
SL = NN2()
STP = NN2()
STP2 = NN2()

joueur = False
size_screen = 3.5
hauteur = 2400/size_screen
largeur = 1080/size_screen
nv_sol = int(120/size_screen)

pygame.init()
pygame.display.set_caption("Run")
screen = pygame.display.set_mode((largeur, hauteur))
background = pygame.transform.scale(pygame.image.load('Fond.jpg'), (largeur, hauteur))

player = pygame.image.load('gps.png')
player = pygame.transform.scale(player, (int(100/size_screen), int(100/size_screen)))

player_Feu = pygame.image.load('gps_Feu.png')
player_Feu = pygame.transform.scale(player_Feu, (int(100/size_screen), int(100/size_screen)))


def dif(a, b):
	if a > b:
		if a - b > 180:
			a -= 360
		a += 360
		b += 360
		return b - a
	else:
		if b - a > 180:
			b -= 360
		a += 360
		b += 360
		return b - a


def no(ro):
	return 360-ro

def Jcos(degrees):
	radians = math.radians(degrees)
	result = math.cos(radians)
	return result

def Jarctan(x):
	return math.degrees(math.atan(x))

#
DCT.sinits(1, 4, 4, 1)
VTS.sinits(1, 5, 5, 1)
SL.sinits(1, 5, 5, 1)
STP.sinits(2, 4, 4, 1)
STP2.sinits(2, 4, 4, 1)
#
G = direction = M = feu = sol = SlM = 0
myfont = pygame.font.SysFont("Arial", int(64/size_screen))

gravity = 1
gravity = gravity/100

victoire = defaite = past = 0


while True:
	y = int(1200/size_screen)
	x = int(540/size_screen)
	rotate = randint(1, 360)
	v_y = randint(0,10)
	v_x = randint(-10,10)
	v = 200
	pygame.event.clear()
	G += 1
	time.sleep(0.2)
	pygame.event.clear()
	while pygame.mouse.get_pressed() == (False,False,False) or joueur == True:
		time.sleep(0.005)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_s:
					time.sleep(10)


		v_x = int(v_x*1000)/1000
		v_y -= gravity/size_screen
		v_y = int(v_y*1000)/1000
		x += v_x/size_screen
		y -= v_y/size_screen
		vitesse = (v_y**2 + v_x**2)**0.5
		vitesse = int(vitesse*100)/100
		screen.blit(background, (0, 0))


		sol = int((hauteur-nv_sol)-y)
	
		
		label2 = myfont.render(str(vitesse)+ ' m/s     |  ' + str(v_y) + ' | ' + str(v_x)+ ' | ' + str(sol), 1, (0,255,0))
		screen.blit(label2, (int(100/size_screen), 0))
		
		
		label2 = myfont.render(str(rotate)+'   | ' + str(direction)+ '   |  '+ str(M), 1, (255, 0, 0))
		screen.blit(label2, (int(400/size_screen), int(100/size_screen)))
		
		
		label2 = myfont.render('Gen  '+str(G)+'     '+str(SlM), 1, (0, 0, 255))
		screen.blit(label2, (int(50/size_screen), int(200/size_screen)))
		if y < 0:
			label2 = myfont.render('Aperçu:', 1, (255, 255, 255))
			screen.blit(label2, ((int(540/size_screen), int(int(1200/size_screen)-int(75/size_screen)))))





		if v_x != 0 and v_y != 0:
			if v_x > 0 and v_y > 0:
				direction = Jarctan(v_x/v_y)
			elif v_x > 0 and v_y < 0:
				direction = (Jarctan(v_y*-1/v_x))+90
			elif v_x < 0 and v_y < 0:
				direction = (Jarctan(v_x*-1/v_y*-1))+180
			elif v_x < 0 and v_y > 0:
				direction = (Jarctan(v_y/v_x*-1))+270
		else:
			if v_y > 0 and v_x == 0:
				direction = 360
			elif v_y < 0 and v_x == 0:
				direction = 180
				
			elif v_x > 0 and v_y == 0:
				direction = 90
			elif v_x < 0 and v_y == 0:
				direction = 270
		direction = (int(direction) + 180) % 360
		if direction == 0:
			direction = 360
			
			
		if feu == True:
			if y < 0:
				screen.blit(pygame.transform.rotate(player_Feu, no(rotate)), (int(540/size_screen), int(1200/size_screen)))
			else:
				screen.blit(pygame.transform.rotate(player_Feu, no(rotate)), (x, y))


			if rotate > 270:
				v_x -= (Jcos(rotate - 270)*M)/size_screen
				v_y += (Jcos(90 - (rotate -270))*M)/size_screen


			elif rotate > 180:
				v_y -= (Jcos(rotate-180)*M)/size_screen
				v_x -= (Jcos(90 - (rotate-180))*M)/size_screen

			elif rotate > 90:
				v_x += (Jcos(rotate-90)*M)/size_screen
				v_y -= (Jcos(90 - (rotate-90))*M)/size_screen

				
				
			elif rotate > 0:
				v_y += (Jcos(rotate)*M)/size_screen
				v_x += (Jcos(90 - rotate)*M)/size_screen
	
			
			#90
			#180
			#270
			#360
		else:
			if y < 0:
				screen.blit(pygame.transform.rotate(player, no(rotate)), (int(540/size_screen), int(1200/size_screen)))

			else:
				screen.blit(pygame.transform.rotate(player, no(rotate)), (x, y))
	
		pygame.display.flip()
		
		if x > int(1080/size_screen):
			x = 0
		elif x < 0:
			x = int(1080/size_screen)
		
		if joueur == True:
			if pygame.mouse.get_pressed() != (False,False,False) and pygame.mouse.get_pos()[1] > int(int(2400/2)/size_screen):
				feu = True
				M = 0.1
			else:
				feu = False
				M = 0
			if pygame.mouse.get_pressed() != (False,False,False) and pygame.mouse.get_pos()[1] < int(int(2400/2)/size_screen) and pygame.mouse.get_pos()[0] < int(int(1080/2)/size_screen):
				rotate -= 1	
			if pygame.mouse.get_pressed() != (False,False,False) and pygame.mouse.get_pos()[1] < int(int(2400/2)/size_screen) and pygame.mouse.get_pos()[0] > int(int(1080/2)/size_screen):
				rotate += 1
				
				
		else:
			if v_x > 0:
				temp = v_x
			else:
				temp = v_x*-1



			STP.u = np.array(([1], [1], [0], [0], [1], [0]), dtype=float)
			STP.x_entrer = np.array(([7,200], [2, 100], [1, 100], [1, 500], [12, 1000], [0.5, 400], [temp, sol]), dtype=float)
			output = float(STP.output(STP.x_entrer, STP.u, 6, False)[0])
			if output > 0.7:
				if v_x > 0:
					SlM =  dif(rotate,no(90))
				else:
					SlM =  dif(rotate,no(270))
			else:
				SlM =  dif(rotate,direction)
			DCT.u = np.array(([0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1]), dtype=float)
			DCT.x_entrer = np.array(([-1], [1], [-3], [3], [-6], [6], [-12], [12], [-20], [20], [-35], [35], [-45], [45], [-60], [60], [SlM]), dtype=float)
			output = DCT.output(DCT.x_entrer , DCT.u, 16, False)
			if output > 0.6:
				rotate += 1
			elif output < 0.4:
				rotate -= 1
			


			STP2.u = np.array(([0], [0], [0], [1], [1], [1], [1], [1], [1]), dtype=float)
			STP2.x_entrer = np.array(([-0.5, 100], [-0.5, 40], [-0.5, 2], [-1, 100], [-1, 40], [-1, 2], [5, 100], [10, 5], [-1, 100], [v_y, sol]), dtype=float)
			output_2 = float(STP2.output(STP2.x_entrer, STP2.u, 9, False)[0])
			print(int(output_2*10000)/10000)

			VTS.u = np.array(([0.01], [0.1], [0.2], [0.3], [0.8], [1], [1]), dtype=float)
			VTS.x_entrer = np.array(([0.5], [1], [3], [5], [7], [10], [30], [vitesse]), dtype=float)
			output = float(VTS.output(VTS.x_entrer, VTS.u, 7, False)[0])

			SL.u = np.array(([0.05], [0.1], [0.2], [0.3], [0.3], [0.4], [0.5], [0.80]), dtype=float)
			SL.x_entrer = np.array(([400], [300], [200], [100], [50], [10], [3], [1], [sol]), dtype=float)
			multi = float(SL.output(SL.x_entrer,SL.u, 8, False)[0])

			feu = True
			if output_2 > 0.5:
				M = output*multi
			else:
				M = output*(multi*(output_2/10))
				feu = False
				
			
		if  y > hauteur-nv_sol:
			print(rotate, vitesse)
			if vitesse < 1 and rotate > 350 or rotate < 10:
				vitesse = 0
				v_y = v_x = 0
				victoire += 1
				time.sleep(4)
				break
			else:
				defaite += 1
				time.sleep(4)
				break
	
			
		if rotate < 1:
			rotate = 360	
		if rotate > 360:
			rotate = 1

	
	
	#1080, 2400
	if joueur == False:
		print('________________')
		print('	Generation:',G)
		print('DCT')
		output = DCT.output(DCT.x_entrer , DCT.u, 16, True)
		print('VST')
		output = VTS.output(VTS.x_entrer , VTS.u, 7, True)
		print('SL')
		output = SL.output(SL.x_entrer , SL.u, 8, True)
		print('STP')
		output = STP.output(STP.x_entrer , STP.u, 5, True)
		print('STP2')
		output = STP2.output(STP2.x_entrer , STP2.u, 9, True)
		print('Victoire',victoire,'	defaite',defaite)
		print('Past:',G - (victoire+defaite))
		print('________________')
		print()
		DCT.train(DCT.x_entrer, DCT.u, v, 16)
		VTS.train(VTS.x_entrer, VTS.u, v, 7)
		SL.train(SL.x_entrer, SL.u, v, 8)
		STP.train(STP.x_entrer, STP.u, v, 6)
		STP2.train(STP2.x_entrer, STP2.u, v, 9)