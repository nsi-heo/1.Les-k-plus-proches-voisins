#Affichage avec la bibliothèque graphique intégrée à Notebook
#Affichage avec la bibliothèque graphique GTK
import pandas
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from math import pi
from sklearn.neighbors import KNeighborsClassifier

ratio = 1 #2.8
rayon = 0.8 #0.9
iris=pandas.read_csv("iris.csv")
x=iris["petal_length"]
y=iris["petal_width"]
lab=iris["species"]
circle1 = plt.Circle((2.5, 0.75), 0.84, facecolor = None, edgecolor='k',fill=False)
#Ellipse
u=2.5     #x-position of the center
v=0.75    #y-position of the center
a=1.017     #radius on the x-axis   1.056
b=a/ratio    #radius on the y-axis a/b=2.4
t = np.linspace(0, 2*pi, 100)

#fig, ax = plt.subplots(2,figsize=(20,20))
#fig, ax = plt.subplots(figsize=(60,60))


"""

ax[0].set_aspect(ratio)  # je choisi le ratio DX/DY pour les echelles des axes
ax[0].add_artist(Ellipse((2.5, 0.75), rayon*ratio, rayon, color='yellow',alpha=0.1))
#plt.figure(figsize=(10,10))

ax[0].plot( u+a*np.cos(t) , v+b*np.sin(t),color='k' )


ax[0].add_artist(circle1)
ax[0].arrow(2.5, 0.75, 0.4, 0.28, head_width=0.05, head_length=0.1, fc='k', ec='k')
ax[0].arrow(2.5, 0.75, 0.67, 0.22, head_width=0.05, head_length=0.1, fc='k', ec='k')
ax[0].arrow(2.5, 0.75, -0.75, -0.14, head_width=0.05, head_length=0.1, fc='k', ec='k')
#ax.arrow(2.5, 0.75, -0.5, -0.25, head_width=0.05, head_length=0.1, fc='k', ec='k')
#ax.arrow(2.5, 0.75, -0.5, -0.48, head_width=0.05, head_length=0.1, fc='k', ec='k')

#ax.axis('equal')
ax[0].scatter(x[lab == 0], y[lab == 0], color='g', label='setosa',alpha=0.3)
ax[0].scatter(x[lab == 1], y[lab == 1], color='r', label='virginica',alpha=0.3)
ax[0].scatter(x[lab == 2], y[lab == 2], color='b', label='versicolor',alpha=0.3)
ax[0].scatter([2.5],[0.75],marker='X',color='k',label='inconnu',s = 205)
ax[0].legend()
#plt.show()
"""

iris = iris.sample(frac=1,random_state=None)



iris.reset_index(drop = True, inplace = True)
iris_test = iris.loc[:29][:]    #jeu de 30 fleurs à tester
iris_train = iris.loc[30:][:]   #jeu de 120 fleurs pour s'entrainer


#######################################################

x=iris_train["petal_length"]
y=iris_train["petal_width"]
lab=iris_train["species"]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(ratio)  # je choisi le ratio DX/DY pour les echelles des axes
ax.scatter(x[lab == 0], y[lab == 0], color='g', label='setosa',alpha=0.3)
ax.scatter(x[lab == 1], y[lab == 1], color='r', label='virginica',alpha=0.3)
ax.scatter(x[lab == 2], y[lab == 2], color='b', label='versicolor',alpha=0.3)
ax.scatter([2.5],[0.75],marker='X',color='k',label='inconnu',s = 25)
ax.scatter([4.85],[1.68],marker='X',color='k',label='inconnu',s = 25)
ax.legend()
ax.plot( u+a*np.cos(t) , v+b*np.sin(t),color='k' )

plt.show()

##############################################################
test_limit =0
if test_limit == 1:
    iris_test = pandas.DataFrame({'petal_length': [4.85, 2.5, 4.5,5.02,2], 'petal_width': [1.68, 0.75, 1.7,1.49,1], 'species': [1, 1, 2,2,0]}, columns = ['petal_length', 'petal_width', 'species'])


iris_train['species'].replace([0,1,2],['setosa','virginica', 'versicolor'],inplace=True)
iris_test['species'].replace([0,1,2],['setosa','virginica', 'versicolor'],inplace=True)
iris_test_species_record=iris_test['species']
iris_test.drop('species',axis=1,inplace=True)




def distance(pt1,pt2):
    """
    Description :Calcule la distance entre deux points en tenant compte du ratio entre les deux échelles de valeurs

    paramètres :
        - pt1 : tuple contenant x1 et y1
        - pt2 : tuple contenant x2 et y2

    retour :
        - Distance entre deux points en tenant compte du ratio
    """
    return (sqrt(((pt1[0]-pt2[0])/ratio)**2+(pt1[1]-pt2[1])**2))    #Distance entre les deux points en tenant
                                                                    #compte du ratio


def plus_proche_voisin(pt,train,k):
    """

    Paramètres :
        - k : nombre de voisins les plus proches

    """
    distances = []

    for ligne_train in train.itertuples():

        dist = distance((ligne_train.petal_length,ligne_train.petal_width),pt)
        distances.append(dist)


    train['dist'] = distances
    distances = iris_train[['dist','species']]
    distances = distances.sort_values(by='dist',ascending = True)
    head = distances.head(k)
    result = head['species'].value_counts()
    #print(head)

    return result.index[0] #renvoie l'index de la valeur la plus élevée (fréquence la plus élevée)



def eprouver_test(test,k):
    previsions = []
    for ligne_test in test.itertuples():
        L_test = ligne_test.petal_length
        l_test = ligne_test.petal_width
        pt=(L_test,l_test)

        result = plus_proche_voisin(pt,iris_train,k)
        previsions.append(result) #



    return (previsions)




def score_knn(k):
    score = 0
    previsions = eprouver_test(iris_test,k)
    print(previsions)
    taille_jeu_test = shape(iris_test)[0]
    for i in range(taille_jeu_test):
        if iris_test_species_record[i]==previsions[i]:
            score +=1
    return score/taille_jeu_test*100



k=3
print('score : ',score_knn(k),' %')

pt = (4.85,1.68)
print(plus_proche_voisin(pt,iris_train,k))
#ax[0].plot( pt[0] , pt[1],color='k',marker='*')


pt = (2.5,0.75)
print(plus_proche_voisin(pt,iris_train,k))
#ax[0].plot( pt[0] , pt[1],color='k',marker='*')


print(iris_test_species_record)
#print(iris_train)





y= iris_train[['species']]
x = iris_train[['petal_length','petal_width']]
y=y.values.ravel()  #pour formater en un tableau à 1d


model=KNeighborsClassifier(n_neighbors=3)

model.fit(x,y)
score = model.score(x,y) #pourcentage de pertinence du modèle

print('pertinence modèle', score*100,' %')


test= np.array([4.85,1.68]).reshape(1,2)
print(model.predict(test))
test= np.array([2.5,0.75]).reshape(1,2)
print(model.predict(test))












