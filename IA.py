import numpy as np


class NN0(object):
	def sinits(self, inputSize, outputSize):
#Nos paramètres
		self.inputSize = inputSize # Nombre de neurones d'entrer
		self.outputSize = outputSize # Nombre de neurones de sortie
#Nos poids
		self.W1 = np.random.randn(self.inputSize,self.outputSize) #Matrice de poids entre les neurones d'entrer et les neurones de sorties
  #Fonction de propagation avant
	def forward(self, X):
		self.z = np.dot(X, self.W1) # Multiplication matricielle entre les valeurs d'entrer et les valeurs de sortie
		o = self.sigmoid(self.z) # Application de la fonction d'activation, et obtention de notre valeur de sortie final
		return o

  # Fonction d'activation
	def sigmoid(self, s):
		return 1/(1+np.exp(-s))

  # Dérivée de la fonction d'activation
	def sigmoidPrime(self, s):
		return s * (1 - s)

  #Fonction de rétropropagation
	def backward(self, X, y, o):
		self.o_error = y - o # Calcul de l'erreur
		self.o_delta = self.o_error*self.sigmoidPrime(o) # Application de la dérivée de la sigmoid à cette erreur
		self.W1 += X.T.dot(self.o_delta) # On ajuste nos poids W1


#Fonction d'entrainement
	def train(self, X, u, toure, nb):
		self.u = np.array((u), dtype=float)
		x_entrer = np.array((X), dtype=float)
		
		x_entrer = x_entrer/np.amax(x_entrer, axis = 0)
		self.X = np.split(x_entrer, [nb])[0]
		xPrediction = np.split(x_entrer, [nb])[1]
		t = 0
		while t != toure:
			t += 1
			o = self.forward(self.X)
			self.backward(self.X, self.u, o)

	def output(self, X, u, nb, vu):
		self.u = np.array((u), dtype=float)
		x_entrer = np.array((X), dtype=float)
		
		x_entrer = x_entrer/np.amax(x_entrer, axis = 0)
		self.X = np.split(x_entrer, [nb])[0]
		xPrediction = np.split(x_entrer, [nb])[1]
		output = np.array(self.forward(xPrediction))
		if vu == True:
			print('Sortie actuelle \n' + str(self.u))
			print('Sortie prédite \n' + str(np.matrix.round(self.forward(X),2)))
			print('\n')
		return output







class NN1(object):
	def sinits(self, inputSize, hiddenSize, outputSize):
#Nos paramètres
		self.inputSize = inputSize # Nombre de neurones d'entrer
		self.outputSize = outputSize # Nombre de neurones de sortie
		self.hiddenSize = hiddenSize # Nombre de neurones cachés
#Nos poids
		self.W1 = np.random.randn(self.inputSize,self.hiddenSize) # (2x3) Matrice de poids entre les neurones d'entrer et cachés
		self.W2 = np.random.randn(self.hiddenSize, 	self.outputSize) # (3x1) Matrice de poids entre les neurones cachés et sortie
#Fonction de propagation avant
  #Fonction de propagation avant
	def forward(self, X):
		self.z = np.dot(X, self.W1) # Multiplication matricielle entre les valeurs d'entrer et les poids W1
		self.z2 = self.sigmoid(self.z) # Application de la fonction d'activation (Sigmoid)
		self.z3 = np.dot(self.z2, self.W2) # Multiplication matricielle entre les valeurs cachés et les poids W2
		o = self.sigmoid(self.z3) # Application de la fonction d'activation, et obtention de notre valeur de sortie final
		return o

  # Fonction d'activation
	def sigmoid(self, s):
		return 1/(1+np.exp(-s))

  # Dérivée de la fonction d'activation
	def sigmoidPrime(self, s):
		return s * (1 - s)

  #Fonction de rétropropagation
	def backward(self, X, y, o):
		self.o_error = y - o # Calcul de l'erreur
		self.o_delta = self.o_error*self.sigmoidPrime(o) # Application de la dérivée de la sigmoid à cette erreur
		self.z2_error = self.o_delta.dot(self.W2.T) # Calcul de l'erreur de nos neurones cachés 
		self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # Application de la dérivée de la sigmoid à cette erreur

		self.W1 += X.T.dot(self.z2_delta) # On ajuste nos poids W1
		self.W2 += self.z2.T.dot(self.o_delta) # On ajuste nos poids W2

#Fonction d'entrainement
	def train(self, X, u, toure, nb):
		self.u = np.array((u), dtype=float)
		x_entrer = np.array((X), dtype=float)
		
		x_entrer = x_entrer/np.amax(x_entrer, axis = 0)
		self.X = np.split(x_entrer, [nb])[0]
		xPrediction = np.split(x_entrer, [nb])[1]
		t = 0
		while t != toure:
			t += 1
			o = self.forward(self.X)
			self.backward(self.X, self.u, o)

	def output(self, X, u, nb, vu):
		self.u = np.array((u), dtype=float)
		x_entrer = np.array((X), dtype=float)
		
		x_entrer = x_entrer/np.amax(x_entrer, axis = 0)
		self.X = np.split(x_entrer, [nb])[0]
		xPrediction = np.split(x_entrer, [nb])[1]
		output = np.array(self.forward(xPrediction))
		if vu == True:
			print('Sortie actuelle \n' + str(self.u))
			print('Sortie prédite \n' + str(np.matrix.round(self.forward(X),2)))
			print('\n')
		return output



##
class NN2(object):
	def sinits(self, inputSize, hiddenSize, hiddenSize_2, outputSize):
#Nos paramètres
		self.inputSize = inputSize # Nombre de neurones d'entrer
		self.outputSize = outputSize # Nombre de neurones de sortie
		self.hiddenSize = hiddenSize # Nombre de neurones cachés
		self.hiddenSize_2 = hiddenSize_2# Nombre de neurones cachés2

#Nos poids

		self.W1 = np.random.randn(self.inputSize,self.hiddenSize) # (4x8) Matrice de poids entre les neurones d'entrer et cachés
		self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize_2) # (8x6) Matrice de poids entre les neurones cachés et la 2 eme couche
		self.W3 = np.random.randn(self.hiddenSize_2, self.outputSize) # (8x6) Matrice de poids entre les neurones cachés2 et la 2 eme couche

		
#Fonction de propagation avant
	def forward(self, X):
		self.z = np.dot(X, self.W1) # Multiplication matricielle entre les valeurs d'entrer et les poids W1
		self.z2 = self.sigmoid(self.z) # Application de la fonction d'activation (Sigmoid)
		
		self.z3 = np.dot(self.z2, self.W2) # Multiplication matricielle entre les valeurs cachés et les poids W2
		self.z4 = self.sigmoid(self.z3) # Application de la fonction d'activation (Sigmoid)
		
		self.z5 = np.dot(self.z4, self.W3) # Multiplication matricielle entre les valeurs cachés et les poids W3
		o = self.sigmoid(self.z5) # Application de la fonction d'activation, et obtention de notre valeur de sortie final
		return o
		
# Fonction d'activation
	def sigmoid(self, s):
		return 1/(1+np.exp(-s))
# Dérivée de la fonction d'activation

	def sigmoidPrime(self, s):
		return s * (1 - s)
		
#Fonction de rétropropagation
	def backward(self, X, u, o):
		self.o_error = u - o # Calcul de l'erreur
		self.o_delta = self.o_error*self.sigmoidPrime(o) # Application de la dérivée de la sigmoid à cette erreur
		#W3
		self.z4_error = self.o_delta.dot(self.W3.T) # Calcul de l'erreur de nos neurones cachés
		self.z4_delta = self.z4_error*self.sigmoidPrime(self.z4) # Application de la dérivée de la sigmoid à cette erreur
		#W2
		self.z2_error = self.z4_delta.dot(self.W2.T) # Calcul de l'erreur de nos neurones cachés
		self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # Application de la dérivée de la sigmoid à cette erreur
		
		#fin
		self.W1 += X.T.dot(self.z2_delta) # On ajuste nos poids W1
		self.W2 += self.z2.T.dot(self.z4_delta) # On ajuste nos poids W2
		self.W3 += self.z4.T.dot(self.o_delta) # On ajuste nos poids W3
#Fonction d'entrainement
	def train(self, X, u, toure, nb):
		self.u = np.array((u), dtype=float)
		x_entrer = np.array((X), dtype=float)
		
		x_entrer = x_entrer/np.amax(x_entrer, axis = 0)
		self.X = np.split(x_entrer, [nb])[0]
		xPrediction = np.split(x_entrer, [nb])[1]
		t = 0
		while t != toure:
			t += 1
			o = self.forward(self.X)
			self.backward(self.X, self.u, o)

	def output(self, X, u, nb, vu):
		self.u = np.array((u), dtype=float)
		x_entrer = np.array((X), dtype=float)
		
		x_entrer = x_entrer/np.amax(x_entrer, axis = 0)
		self.X = np.split(x_entrer, [nb])[0]
		xPrediction = np.split(x_entrer, [nb])[1]
		output = np.array(self.forward(xPrediction))
		if vu == True:
			print('Sortie actuelle \n' + str(self.u))
			print('Sortie prédite \n' + str(np.matrix.round(self.forward(X),2)))
			print('\n')
		return output

		
		




class NN3(object):
	def sinits(self, inputSize, hiddenSize, hiddenSize_2, hiddenSize_3, outputSize):
#Nos paramètres
		self.inputSize = inputSize # Nombre de neurones d'entrer
		self.outputSize = outputSize # Nombre de neurones de sortie
		self.hiddenSize = hiddenSize # Nombre de neurones cachés
		self.hiddenSize_2 = hiddenSize_2# Nombre de
		self.hiddenSize_3 = hiddenSize_3# Nombre de
#Nos poids

		self.W1 = np.random.randn(self.inputSize,self.hiddenSize) # (4x8) Matrice de poids entre les neurones d'entrer et cachés
		self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize_2) # (8x6) Matrice de poids entre les neurones cachés et la 2 eme couche
		self.W3 = np.random.randn(self.hiddenSize_2, self.hiddenSize_3) # (8x6) Matrice de poids entre les neurones cachés2 et la 2 eme couche
		self.W4 = np.random.randn(self.hiddenSize_3, self.outputSize) # (6x2) Matrice de poids entre les neurones cachés2 et sortie
		
#Fonction de propagation avant
	def forward(self, X):
		self.z = np.dot(X, self.W1) # Multiplication matricielle entre les valeurs d'entrer et les poids W1
		self.z2 = self.sigmoid(self.z) # Application de la fonction d'activation (Sigmoid)
		self.z3 = np.dot(self.z2, self.W2) # Multiplication matricielle entre les valeurs cachés et les poids W2
		self.z4 = self.sigmoid(self.z3)#Sigmoid
		
		self.z5 = np.dot(self.z4, self.W3) # Multiplication matricielle entre les valeurs cachés et les poids W3
		self.z6 = self.sigmoid(self.z5)#Sigmoid
		
		self.z7 = np.dot(self.z6, self.W4)#Multiplic1tion entre couche3 et poid W4
		o = self.sigmoid(self.z7) # Application de la fonction d'activation, et obtention de notre valeur de sortie final
		return o
# Fonction d'activation
	def sigmoid(self, s):
		return 1/(1+np.exp(-s))
# Dérivée de la fonction d'activation
	def sigmoidPrime(self, s):
		return s * (1 - s)
#Fonction de rétropropagation
	def backward(self, X, u, o):
		self.o_error = u - o # Calcul de l'erreur
		self.o_delta = self.o_error*self.sigmoidPrime(o) # Application de la dérivée de la sigmoid à cette erreur
		#W4
		self.z6_error = self.o_delta.dot(self.W4.T) # Calcul de l'erreur de nos neurones cachés
		self.z6_delta = self.z6_error*self.sigmoidPrime(self.z6) # Application de la dérivée de la sigmoid à cette erreur
		#W3
		self.z4_error = self.z6.dot(self.W3.T) # Calcul de l'erreur de nos neurones cachés
		self.z4_delta = self.z4_error*self.sigmoidPrime(self.z4) # Application de la dérivée de la sigmoid à cette erreur
		#W2
		self.z2_error = self.z4_delta.dot(self.W2.T) # Calcul de l'erreur de nos neurones cachés
		self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # Application de la dérivée de la sigmoid à cette erreur
		
		#fin
		self.W1 += X.T.dot(self.z2_delta) # On ajuste nos poids W1
		self.W2 += self.z2.T.dot(self.z4_delta) # On ajuste nos poids W2
		self.W3 += self.z4.T.dot(self.z6_delta) # On ajuste nos poids W3
		self.W4 += self.z6.T.dot(self.o_delta) # On ajuste nos poids W4
	def train(self, X, u, toure, nb):
		self.u = np.array((u), dtype=float)
		x_entrer = np.array((X), dtype=float)
		
		x_entrer = x_entrer/np.amax(x_entrer, axis = 0)
		self.X = np.split(x_entrer, [nb])[0]
		xPrediction = np.split(x_entrer, [nb])[1]
		t = 0
		while t != toure:
			t += 1
			o = self.forward(self.X)
			self.backward(self.X, self.u, o)

	def output(self, X, u, nb):
		self.u = np.array((u), dtype=float)
		x_entrer = np.array((X), dtype=float)
		
		x_entrer = x_entrer/np.amax(x_entrer, axis = 0)
		self.X = np.split(x_entrer, [nb])[0]
		xPrediction = np.split(x_entrer, [nb])[1]
		output = np.array(self.forward(xPrediction))
		return output



class NN4(object):
	def sinits(self, inputSize, hiddenSize, hiddenSize_2, hiddenSize_3, hiddenSize_4, outputSize):
#Nos paramètres
		self.inputSize = inputSize # Nombre de neurones d'entrer
		self.outputSize = outputSize # Nombre de neurones de sortie
		self.hiddenSize = hiddenSize # Nombre de neurones cachés
		self.hiddenSize_2 = hiddenSize_2# Nombre de
		self.hiddenSize_3 = hiddenSize_3# Nombre de
		self.hiddenSize_4 = hiddenSize_4# Nombre de
#Nos poids

		self.W1 = np.random.randn(self.inputSize,self.hiddenSize) # (4x8) Matrice de poids entre les neurones d'entrer et cachés
		self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize_2) # (8x6) Matrice de poids entre les neurones cachés et la 2 eme couche
		self.W3 = np.random.randn(self.hiddenSize_2, self.hiddenSize_3) # (8x6) Matrice de poids entre les neurones cachés2 et la 2 eme couche
		self.W4 = np.random.randn(self.hiddenSize_3, self.hiddenSize_4) # (6x2) Matrice de poids entre les neurones cachés2 et sortie
		self.W5 = np.random.randn(self.hiddenSize_4, self.outputSize) # (6x2) Matrice de poids entre les neurones cachés2 et sortie
		
#Fonction de propagation avant
	def forward(self, X):
		self.z = np.dot(X, self.W1) # Multiplication matricielle entre les valeurs d'entrer et les poids W1
		self.z2 = self.sigmoid(self.z) # Application de la fonction d'activation (Sigmoid)

		self.z3 = np.dot(self.z2, self.W2) # Multiplication matricielle entre les valeurs cachés et les poids W2
		self.z4 = self.sigmoid(self.z3)#Sigmoid
		
		self.z5 = np.dot(self.z4, self.W3) # Multiplication matricielle entre les valeurs cachés et les poids W3
		self.z6 = self.sigmoid(self.z5)#Sigmoid

		self.z7 = np.dot(self.z6, self.W4) # Multiplication matricielle entre les valeurs cachés et les poids W3
		self.z8 = self.sigmoid(self.z7)#Sigmoid
		
		self.z9 = np.dot(self.z8, self.W5)#Multiplic1tion entre couche3 et poid W4
		o = self.sigmoid(self.z9) # Application de la fonction d'activation, et obtention de notre valeur de sortie final
		return o
# Fonction d'activation
	def sigmoid(self, s):
		return 1/(1+np.exp(-s))
# Dérivée de la fonction d'activation
	def sigmoidPrime(self, s):
		return s * (1 - s)
#Fonction de rétropropagation
	def backward(self, X, u, o):
		self.o_error = u - o # Calcul de l'erreur
		self.o_delta = self.o_error*self.sigmoidPrime(o) # Application de la dérivée de la sigmoid à cette erreur
		#W5
		self.z8_error = self.o_delta.dot(self.W5.T) # Calcul de l'erreur de nos neurones cachés
		self.z8_delta = self.z8_error*self.sigmoidPrime(self.z8) # Application de la dérivée de la sigmoid à cette erreur
		#W4
		self.z6_error = self.z8.dot(self.W4.T) # Calcul de l'erreur de nos neurones cachés
		self.z6_delta = self.z6_error*self.sigmoidPrime(self.z6) # Application de la dérivée de la sigmoid à cette erreur
		#W3
		self.z4_error = self.z6.dot(self.W3.T) # Calcul de l'erreur de nos neurones cachés
		self.z4_delta = self.z4_error*self.sigmoidPrime(self.z4) # Application de la dérivée de la sigmoid à cette erreur
		#W2
		self.z2_error = self.z4_delta.dot(self.W2.T) # Calcul de l'erreur de nos neurones cachés
		self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # Application de la dérivée de la sigmoid à cette erreur
		
		#fin
		self.W1 += X.T.dot(self.z2_delta) # On ajuste nos poids W1
		self.W2 += self.z2.T.dot(self.z4_delta) # On ajuste nos poids W2
		self.W3 += self.z4.T.dot(self.z6_delta) # On ajuste nos poids W3
		self.W4 += self.z6.T.dot(self.z8_delta) # On ajuste nos poids W4
		self.W5 += self.z8.T.dot(self.o_delta) # On ajuste nos poids W5
	def train(self, X, u, toure):
		self.u = np.array((u), dtype=float)
		x_entrer = np.array((x), dtype=float)
		
		x_entrer = x_entrer/np.amax(x_entrer, axis = 0)
		self.X = np.split(x_entrer, [nb])[0]
		xPrediction = np.split(x_entrer, [nb])[1]
		t = 0
		while t != toure:
			o = self.forward(X)
			self.backward(X, u, o)

	def output(u,x, nb):
		self.u = np.array((u), dtype=float)
		x_entrer = np.array((x), dtype=float)
		
		x_entrer = x_entrer/np.amax(x_entrer, axis = 0)
		self.X = np.split(x_entrer, [nb])[0]
		xPrediction = np.split(x_entrer, [nb])[1]
		output = np.array(NN.forward(xPrediction))
		return output






class NN5(object):
	def sinits(self, inputSize, hiddenSize, hiddenSize_2, hiddenSize_3, hiddenSize_4, hiddenSize_5, outputSize):
#Nos paramètres
		self.inputSize = inputSize # Nombre de neurones d'entrer
		self.outputSize = outputSize # Nombre de neurones de sortie
		self.hiddenSize = hiddenSize # Nombre de neurones cachés
		self.hiddenSize_2 = hiddenSize_2
		self.hiddenSize_3 = hiddenSize_3
		self.hiddenSize_4 = hiddenSize_4
		self.hiddenSize_5 = hiddenSize_5
#Nos poids

		self.W1 = np.random.randn(self.inputSize,self.hiddenSize) # (4x8) Matrice de poids entre les neurones d'entrer et cachés
		self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize_2) # (8x6) Matrice de poids entre les neurones cachés et la 2 eme couche
		self.W3 = np.random.randn(self.hiddenSize_2, self.hiddenSize_3) # (8x6) Matrice de poids entre les neurones cachés2 et la 2 eme couche
		self.W4 = np.random.randn(self.hiddenSize_3, self.hiddenSize_4) # (6x2) Matrice de poids entre les neurones cachés2 et sortie
		self.W5 = np.random.randn(self.hiddenSize_4, self.hiddenSize_5) # (6x2) Matrice de poids entre les neurones cachés2 et sortie
		self.W6 = np.random.randn(self.hiddenSize_5, self.outputSize) # (6x2) Matrice de poids entre les neurones cachés2 et sortie
		
#Fonction de propagation avant
	def forward(self, X):
		self.z = np.dot(X, self.W1) # Multiplication matricielle entre les valeurs d'entrer et les poids W1
		self.z2 = self.sigmoid(self.z) # Application de la fonction d'activation (Sigmoid)

		self.z3 = np.dot(self.z2, self.W2) # Multiplication matricielle entre les valeurs cachés et les poids W2
		self.z4 = self.sigmoid(self.z3)#Sigmoid
		
		self.z5 = np.dot(self.z4, self.W3) # Multiplication matricielle entre les valeurs cachés et les poids W3
		self.z6 = self.sigmoid(self.z5)#Sigmoid

		self.z7 = np.dot(self.z6, self.W4) # Multiplication matricielle entre les valeurs cachés et les poids W3
		self.z8 = self.sigmoid(self.z7)#Sigmoid

		self.z9 = np.dot(self.z8, self.W5) # Multiplication matricielle entre les valeurs cachés et les poids W3
		self.z10 = self.sigmoid(self.z9)#Sigmoid
		
		self.z11 = np.dot(self.z10, self.W6)#Multiplic1tion entre couche3 et poid W4
		o = self.sigmoid(self.z11) # Application de la fonction d'activation, et obtention de notre valeur de sortie final
		return o
# Fonction d'activation
	def sigmoid(self, s):
		return 1/(1+np.exp(-s))
# Dérivée de la fonction d'activation
	def sigmoidPrime(self, s):
		return s * (1 - s)
#Fonction de rétropropagation
	def backward(self, X, u, o):
		self.o_error = u - o # Calcul de l'erreur
		self.o_delta = self.o_error*self.sigmoidPrime(o) # Application de la dérivée de la sigmoid à cette erreur
		#W6
		self.z10_error = self.o_delta.dot(self.W6.T) # Calcul de l'erreur de nos neurones cachés
		self.z10_delta = self.z10_error*self.sigmoidPrime(self.z10) # Application de la dérivée de la sigmoid à cette erreur
		#W5
		self.z8_error = self.o_delta.dot(self.W5.T) # Calcul de l'erreur de nos neurones cachés
		self.z8_delta = self.z8_error*self.sigmoidPrime(self.z8) # Application de la dérivée de la sigmoid à cette erreur
		#W4
		self.z6_error = self.z8.dot(self.W4.T) # Calcul de l'erreur de nos neurones cachés
		self.z6_delta = self.z6_error*self.sigmoidPrime(self.z6) # Application de la dérivée de la sigmoid à cette erreur
		#W3
		self.z4_error = self.z6.dot(self.W3.T) # Calcul de l'erreur de nos neurones cachés
		self.z4_delta = self.z4_error*self.sigmoidPrime(self.z4) # Application de la dérivée de la sigmoid à cette erreur
		#W2
		self.z2_error = self.z4_delta.dot(self.W2.T) # Calcul de l'erreur de nos neurones cachés
		self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # Application de la dérivée de la sigmoid à cette erreur
		
		#fin
		self.W1 += X.T.dot(self.z2_delta) # On ajuste nos poids W1
		self.W2 += self.z2.T.dot(self.z4_delta) # On ajuste nos poids W2
		self.W3 += self.z4.T.dot(self.z6_delta) # On ajuste nos poids W3
		self.W4 += self.z6.T.dot(self.z8_delta) # On ajuste nos poids W4
		self.W5 += self.z8.T.dot(self.z10_delta) # On ajuste nos poids W5
		self.W6 += self.z10.T.dot(self.o_delta) # On ajuste nos poids W6
	def train(self, X, u, toure):
		self.u = np.array((u), dtype=float)
		x_entrer = np.array((x), dtype=float)
		
		x_entrer = x_entrer/np.amax(x_entrer, axis = 0)
		self.X = np.split(x_entrer, [nb])[0]
		xPrediction = np.split(x_entrer, [nb])[1]
		t = 0
		while t != toure:
			o = self.forward(X)
			self.backward(X, u, o)

	def output(u,x, nb):
		self.u = np.array((u), dtype=float)
		x_entrer = np.array((x), dtype=float)
		
		x_entrer = x_entrer/np.amax(x_entrer, axis = 0)
		self.X = np.split(x_entrer, [nb])[0]
		xPrediction = np.split(x_entrer, [nb])[1]
		output = np.array(NN.forward(xPrediction))
		return output