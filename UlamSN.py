import numpy as np
import matplotlib.pyplot as plt

#Parameters
b=3
x1=-5		#Initial condition
x2=-5
#N=35000 #length of partition
N=20000

#Define the mapping
def T(x):
  return b*(1-np.exp(-x))/(1+np.exp(-x))

def Tinv(x):
  return np.log((1+x/b)/(1-x/b))

def dT(x):
    return 2*b*np.exp(-x)/((1+np.exp(-x))**2)

def ddT(x):
    return 2*b*(np.exp(-x)-1)/((np.exp(-x)+1)**2)




#Value of critical noise strength
extra=np.longdouble(np.sqrt((b-1)**2-1))
Sesp=-np.longdouble(np.log(b-1+extra))+b*(b-2+extra)/(b+extra)

MARK=0
#for S in [4*Sesp/8,5*Sesp/8,6*Sesp/8,7*Sesp/8]:


#for S in [Sesp/2]:      #Hyperbolic
for S in [Sesp]:            #Nonhyperbolic
    mark=0
    
    #We find left and right boundary of MIS
    x1=-4
    x2=-2
    for j in range(100000):
        x2=T(x2)+S
        x1=T(x1)-S
    
    #AC=-1/(2*np.log(dT(x2)))    #Hyperbolic (comment otherwise)
    AC= 4/ddT(x2)
    
    #We define the partition of the MIS
    X=np.linspace(x1,x2,N+1)
    #partition length
    h=X[1]-X[0]
    
    #Define the transition probability of falling in [a,y] starting at x
    def TP(x,a,y):
        if y<T(x)-S:
            return 0
        elif T(x)-S<=y and y<=T(x)+S:
            return (1/(2*S))*(y-np.maximum(a,T(x)-S))
        else:
            return (1/(2*S))*(T(x)+S-np.minimum(a,T(x)+S))
    
    #We define the stochastic matrix
    Q=np.zeros((N,N))
    for i in range(N):
        print(MARK+mark/N)
        for j in range(N):
            Q[i,j]=(1/4)*(TP(X[i+1],X[j],X[j+1])+2*TP((X[i]+X[i+1])/2,X[j],X[j+1])+TP(X[i],X[j],X[j+1]))
        
        mark=mark+1
    
    Qt=Q.transpose()
    
    Stationary=np.zeros(N)
    Stationary[0]=1
    
    for i in range(700):
        Stationary= Qt.dot(Stationary)
    
    Stationary=(1/(h*np.linalg.norm(Stationary,ord=1)))*Stationary
    
    rescaleST=np.log(-np.log(Stationary[1500:]))
    #rescaleX=np.log(-np.log(x2-X[1500:N]))  #Hyperbolic
    rescaleX=-np.log(x2-X[1500:N])  #Nonhyperbolic
    #rescaleTheo=np.log(AC)+2*np.log(-np.log(x2-X[1500:N]))  #Hyperbolic
    rescaleTheo=np.log(AC)+np.log(-np.log(x2-X[1500:N])) -np.log(x2-X[1500:N])  #Nonhyperbolic


DIST=x2-X[1500:N]
ST=Stationary[1500:]
QUANT=np.multiply(DIST,np.log(ST))
QUANT=np.multiply(QUANT,np.reciprocal(np.log(DIST)))

LogPhiApp=AC*np.multiply(np.log(DIST),np.reciprocal(DIST))
ERROR=np.absolute(np.log(ST),LogPhiApp)

plt.plot(X[1500:N],QUANT)
plt.plot([x1,x2],[AC,AC], 'r:')
plt.plot([x2,x2],[0,AC+5], 'k:')

#plt.plot(rescaleX,rescaleST,color='tab:green')
#plt.plot(rescaleX,rescaleTheo, 'r:')
plt.show()
