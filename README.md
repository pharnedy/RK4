RK4
===

numerical_solution_of_charge_profile_in_pn_junction


from pylab import*
import scipy.optimize as optimize
#
epsilon_0=8.854e-12 # F/m 
def Eg(z):
  if z <= junction:
		return 1.424 #eV
	if z>junction:
		return 1.798 #eV
def ee0(z):
	if z <= junction:
		return 13.1*8.854e-12 #F/m
	if z>junction:
		return 12.2*8.854e-12 #F/m
def Me(z):
	if z <= junction:
		return 0.0665
	if z>junction:
		return 0.0914
def Mh(z):
	if z <= junction:
		return 0.5
	if z>junction:
		return 0.587
def Na(z):
	if z <= junction:
		return 1E18
	if z > junction:
		return 0
def Nd(z):
	if z <= junction:
		return 0
	if z>junction:
		return 2E17
def Nc(z):
	if z <= junction:
		return 2.51E19*((.0665*(T/300))**1.5)
	if z>junction:
		return 2.51E19*((.0914*(T/300))**1.5)
def Nv(z):
	if z <= junction:
		return 2.51E19*((.5*(T/300))**1.5)
	if z>junction:
		return 2.51E19*((.587*(T/300))**1.5)



#Ea=Ev acceptor energy set eaquel to Ec
#Ed=Ec donor energy set equal to Ev        		 


def ni(z,Nc,Nv,Eg,K,T):
	return (sqrt(Nc(z)*Nv(z)))*exp(-Eg(z)/(2*K*T))
def Ei(z,Ev,Eg,K,T,Mh,Me):
	return ((Ev+Eg(z)+Ev)/2)+((3*K*T)/4)*log(Mh(z)/Me(z))	
def P(z,K,T,Ef,Nc,Nv,Eg,Ev,Mh,Me):
	return ni(z,Nc,Nv,Eg,K,T)*exp((Ei(z,Ev,Eg,K,T,Mh,Me)-Ef)/((K*T)))		
def N(z,K,T,Ef,Ev,Eg):
	return ni(z,Nc,Nv,Eg,K,T)*exp((Ef-Ei(z,Ev,Eg,K,T,Mh,Me))/((K*T)))
def Naa(z,Na,Ef,Ev,K,T):
	return ((Na(z))/(1+(4*exp(Ev-Ef/((K*T))))))		
def Ndd(z,Nd,Ef,K,T,Ev,Eg):
	return (Nd(z)*(1-(1/(1+.5*exp((Ev+Eg(z)-Ef)/((K*T)))))))		

	
def rhoP(Ev):
	rhoP=P(0,K,T,Ef,Nc,Nv,Eg,Ev,Mh,Me)-N(0,K,T,Ef,Ev,Eg)+Ndd(0,Nd,Ef,K,T,Ev,Eg)-Naa(0,Na,Ef,Ev,K,T)
	return rhoP
def rhoN(Ev):
	rhoN=-P(numPoints-1,K,T,Ef,Nc,Nv,Eg,Ev,Mh,Me)+N(numPoints-1,K,T,Ef,Ev,Eg)-Ndd(numPoints-1,Nd,Ef,K,T,Ev,Eg)+Naa(numPoints-1,Na,Ef,Ev,K,T)
	return rhoN



def Ev(z,y,E_material):
	#print z
	if z == 0:
		E_V=E_material
	else:
		E_V=E_material-(q*y)
	return E_V


def ni1(z,Nc,Nv,Eg,K,T):
	return (sqrt(Nc(z)*Nv(z)))*exp(-Eg(z)/(2*K*T))
def Ei1(z,Ev1,Eg,K,T,Mh,Me):
	return ((Ev1+Eg(z)+Ev1)/2)+((3*K*T)/4)*log(Mh(z)/Me(z))	
def P1(z,K,T,Ef,Nc,Nv,Eg,Ev1,Mh,Me):
	return ni1(z,Nc,Nv,Eg,K,T)*exp((Ei1(z,Ev1,Eg,K,T,Mh,Me)-Ef)/((K*T)))		
def N1(z,K,T,Ef,Ev1,Eg):
	return ni1(z,Nc,Nv,Eg,K,T)*exp((Ef-Ei1(z,Ev1,Eg,K,T,Mh,Me))/((K*T)))
def Naa1(z,Na,Ef,Ev1,K,T):
	return ((Na(z))/(1+(4*exp(Ev1-Ef/((K*T))))))		
def Ndd1(z,Nd,Ef,K,T,Ev1,Eg):
	return (Nd(z)*(1-(1/(1+.5*exp((Ev1+Eg(z)-Ef)/((K*T)))))))		


def dtdz(i,P1,N1,Naa1,Ndd1,ee0,Ev1):
	return (-q/ee0(i))*(P1(i,K,T,Ef,Nc,Nv,Eg,Ev1,Mh,Me)+N1(i,K,T,Ef,Ev1,Eg)-Ndd1(i,Nd,Ef,K,T,Ev1,Eg)+((Na(i))/(1+(4*exp((Ev1-Ef)/((K*T)))))))#Naa1(i,Na,Ev,Ef,K,T))##


#  fourth-order Runge-Kutta integrator
def RKThreeD(x1,y1,i,dtdz,dz,Ev1):
#	print Ev1
	k1x = dz * dtdz(i,P1,N1,Naa1,Ndd1,ee0,Ev1)
	k1y = dz * k1x
	
	k2x = dz * dtdz(i+(dz/2),P1,N1,Naa1,Ndd1,ee0,Ev1)+(k1x/2)
	k2y = dz * k2x+(k1y/2)
	
	k3x = dz * dtdz(i+(dz/2) ,P1,N1,Naa1,Ndd1,ee0,Ev1)+ (k2x/2)
	k3y = dz * k3x+ (k2y/2)

	k4x = dz * dtdz((i+dz) ,P1,N1,Naa1,Ndd1,ee0,Ev1)+ (k3x)
	k4y = dz * k3x + (k3y)
	
	x1 += (( k1x + 2.0 * k2x + 2.0 * k3x + k4x ) / 6.0)
	y1 += (( k1y + 2.0 * k2y + 2.0 * k3y + k4y ) / 6.0)

	return x1,y1

#Device=20micron
#junction=10micron

device=200
dz=.01
numPoints=(device/dz)
junction=numPoints/2#material interface

T=300#temperature
q=1.6E-19#electronic charge
K=(1.380648E-23/1.6E-19)#boltzmann constant in eV
Ef=0

E_V=(optimize.bisect(rhoP,-1,1,xtol=1E-100))
print E_V,rhoP(E_V)
E_V1=(optimize.bisect(rhoN,100,-100,xtol=1E-100))
print E_V1,rhoN(E_V1)



x=[]
y=[]

x.append(0.)
y.append(0.)

ev=[]
ev.append(E_V)
for i in range(int(numPoints-1)):
	#ec and ev should be in there
	xt,yt = RKThreeD(x[i],y[i],i,dtdz,dz,ev[i])
	x.append(xt)
	y.append(yt) #potential
	if i <= junction:
		Ev1=Ev(i,yt,E_V)
	if i > junction:
		Ev1=Ev(i,yt,E_V1)
	ev.append(Ev1)

# Plot the trajectory in the phase plane

plot(x)
plot(y)
show()
plot(ev)
show()
































































