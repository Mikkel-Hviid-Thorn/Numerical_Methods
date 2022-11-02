# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:29:06 2021

@authors: A411:
    Andreas Galavits Olsen,
    Emilia Jun Nielsen,
    Esben Hjort Bonnerup,
    Mikkel Hviid Thorn og
    Silje Post Stroem
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

plt.rc('font', size=20)          # Text stoerrelse
plt.rc('axes', titlesize=20)     # Aksetitler stoerrelse
plt.rc('axes', labelsize=20)     # Aksemaerkat stoerrelse
plt.rc('xtick', labelsize=12)    # Aksetal stoerrelse
plt.rc('ytick', labelsize=12)    # Aksetal stoerrelse
plt.rc('legend', fontsize=18)    # Legend stoerrelse
plt.rc('figure', titlesize=25)   # Titel stoerrelse




# Konstanter sat til 1
rho = 1; my = 1; Lz = 1; cp = 1; k = 1;
Nu = 1; f = 1; Ti = 1; Ts = 2; a0 = 2

dT = Ts-Ti

# Funktionen P
P = lambda a,v: dT*rho*cp*np.pi*(a**2)*v*(1-np.exp(-(Nu*k*Lz)/(cp*rho*(a**2)*v)))-(np.pi*rho*f*Lz*a*(v**3))/4

# Partielt afledte
dP_da = lambda a,v: -np.exp(-(Nu*k*Lz)/(cp*rho*v*(a**2)))*(2*np.pi*dT*rho*cp*a*v+(2*np.pi*dT*Nu*k*Lz)/a)+2*np.pi*dT*rho*cp*a*v-(np.pi*rho*f*Lz*(v**3))/4
dP_dv = lambda a,v: -np.exp(-(Nu*k*Lz)/(cp*rho*v*(a**2)))*(np.pi*dT*rho*cp*(a**2)+(np.pi*dT*Nu*k*Lz)/v)+np.pi*dT*rho*cp*(a**2)-(3*np.pi*rho*f*Lz*a*(v**2))/4

def gradient(a,v): # Udregner gradienten af P
    return dP_da(a,v), dP_dv(a,v)

# Konstant v0
v0 = ((16*(dT**2)*cp*Nu*k)/(Lz*rho*(f**2)))**(1/5)
#print(v0)




def plot_region(a_upper,v_upper,N):
    # Numerisk eksperiment, der sammenligner de approksimerede graenser,
    # for hvornaar P er positiv og det omraade hvor P faktisk er positivt.
    A, V = np.linspace(0.001,a_upper,N), np.linspace(0.001,v_upper,N)
    
    P_neg, P_pos = [[],[]], [[],[]]
    for a in A:
        for v in V:
            if P(a,v) >= 0: # Plotter punkter der giver positiv vaerdi med roed
                P_pos[0].append(a)
                P_pos[1].append(v)

            elif P(a,v) < 0: # Plotter punkter der giver negativ vaerdi med blaa
                P_neg[0].append(a)
                P_neg[1].append(v)
    
    plt.plot(P_neg[0],P_neg[1],'o',color='lightgreen', label='Negativt område')
    plt.plot(P_pos[0],P_pos[1],'o',color='lightskyblue', label='Positivt område')
    
    # Indstillingerne for plottet
    plt.xlabel(r'Radiussen $a$'); plt.ylabel(r'Farten $v$')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')


def plot_graenser(a_upper,v_upper,N):
    A = np.linspace(0.001,a_upper,N)
    
    # Estimater for maksimale vaerdier af v
    v_max1 = lambda a: np.sqrt((4*dT*cp*a)/(f*Lz))
    v_max2 = lambda a: ((4*dT*Nu*k)/(rho*f*a))**(1/3)
    
    # Plotter estimateterne
    plt.plot(A,v_max1(A), color='black', label=r'$v=2\sqrt{a}$')
    plt.plot(A,v_max2(A), color='firebrick', label=r'$v=\sqrt[3]{4/a}$')
    
    # Indstillingerne for plottet
    plt.xlim(0, a_upper); plt.ylim(0, v_upper)
    plt.title('Plot over punktmængden med positive værdier')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')


def gradientmetode(a,v,h,N):
    """
    Anvender gradientmetoden.

    Parametre
    ----------
    a : Gaet paa a's vaerdi i maksimum
    v : Gaet paa v's vaerdi i maksimum
    h : Skridtlaengde
    N : Antal iterationer

    Returnerer
    -------
    A,V : punktfoelge
    """
    
    if a<=0 or a>a0 or v<=0 or v>v0: # Tjekker at gaettet er i indenfor bibetingelser
        return print('not legal guess')
    
    A, V = [a], [v]
    for i in range(N):
        dPa, dPv = gradient(a,v) # Udregner gradienten
        length = np.sqrt(dPa**2+dPv**2)
        a_temp = a+h*dPa/length # Danner nyt punkt
        v_temp = v+h*dPv/length
        
        if a_temp <=0: # Overholder graenser for a
            a = 0.001
        elif a_temp > a0:
            a = a0
        else:
            a = a_temp
        
        if v_temp <=0: # Overholder graenser for v
            v = 0.001
        elif v_temp > v0:
            v = v0
        else:
            v = v_temp
        
        A.append(a); V.append(v)
        
        if len(A) > 9: # Stopper hvis de seneste punkter er indenfor en hvis tolerance
            temp_dist = np.sqrt((A[-1]-A[-9])**2+(V[-1]-V[-9])**2)
            if temp_dist < 0.000001:
                break
        
    return A,V




def test_graenser(a_upper,v_upper,N): # Sammenligner teoretiske og reelle graenser
    plt.figure()
    plot_region(a_upper,v_upper,N)
    plot_graenser(a_upper,v_upper,N)
    plt.savefig('test_graenser.png', dpi=1000,bbox_inches='tight')
    
    
def undersoeg_forskellig_start(): # Undersoeger forskellige startpunkter for gradientmetoden
    plt.figure()
    plot_region(a0, v0, 1000) # Plotter det negative og positive omraade
    
    AA = np.linspace(0.25,a0-0.25,5) # Startpunkter
    VV = np.linspace(0.25,v0-0.25,5)
    
    # Farver for punktfoelgerne
    colors = ['maroon','firebrick','r','tomato','indianred',
              'maroon','firebrick','r','tomato','indianred',
              'maroon','firebrick','r','tomato','indianred',
              'maroon','firebrick','r','tomato','indianred',
              'maroon','firebrick','r','tomato','indianred']
    
    a_save, v_save = [], [] # Gemmer de fundne maksimumspunkter
    i = 0
    for a in AA:
        for v in VV: # Anvender gradientmetoden
            A,V = gradientmetode(a,v,0.01,20000)
            
            plt.plot(A[0],V[0],'o',color=colors[i]) # Plotter foerste punkt
            plt.plot(A,V,'--',color=colors[i]) # Plotter punktfoelgen
            plt.plot(A[-1],V[-1],'o',color='black') # Plotter sidste punkt
            
            i += 1
            a_save.append(A[-1]); v_save.append(V[-1]) # Gemmer maksimumspunktet
    
    # Indstillingerne for plottet
    plt.title('Gradientmetoden fra forskellige punkter')
    plt.savefig('undersoeg_start.png', dpi=1000,bbox_inches='tight')
    
    return print(f'a={np.mean(a_save)}', f'v={np.mean(v_save)}', f'sigma_a={np.std(a_save)}', 
                 f'sigma_v={np.std(v_save)}', f'P={P(np.mean(a_save),np.mean(v_save))}',
                 f'gradient={gradient(np.mean(a_save),np.mean(v_save))}')


def a0_test(): # Tester forskellige vaerdier af a0
    global a0
    A0 = [0.02+0.02*i for i in range(301)] # Vaerdier testet
    
    P_values,V_values,A_values = [],[],[]
    for c in A0: # Itererer over a0 vaerdier
        a0 = c
        a,v = gradientmetode(a0/2, v0/2, 0.01, 100000)
        
        # Gemmer resultaterne
        A_values.append(a[-1]); V_values.append(v[-1])
        P_values.append(P(a[-1],v[-1]))
    
    # Plotter tre grafer for a, v og P i maksimumspunktet
    plt.figure()
    plt.plot(A0,A_values,color='firebrick', label=r'$a$-værdi')
    plt.plot(A0,V_values,color='navy', label=r'$v$-værdi')
    plt.plot(A0,P_values,color='forestgreen', label=r'$P$-værdi')
    
    # Indstillingerne for plottet
    plt.xlabel(r'Øvre grænse $a_0$')
    plt.title(r'Forskellige værdier i maksimum ved variation af $a_0$')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.savefig('a0_test.png', dpi=1000,bbox_inches='tight')
    
    # Plot over de partielt afledte
    plt.figure()
    plt.plot(A0,dP_da(np.array(A_values),np.array(V_values)),color='firebrick', label=r'$\frac{\partial P}{\partial a}$')
    plt.plot(A0,dP_dv(np.array(A_values),np.array(V_values)),color='navy', label=r'$\frac{\partial P}{\partial v}$')
    
    # Indstillingerne for plottet
    plt.xlabel(r'Øvre grænse $a_0$')
    plt.title(r'De partielt afledte af $P$ i maksimumspunkterne')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=25)
    plt.savefig('partiel_test.png', dpi=1000,bbox_inches='tight')
    

#test_graenser(3,3,1000)

#undersoeg_forskellig_start()

#a0 = 4
#undersoeg_forskellig_start()

#a0_test()


def snitkurver():
    global a0
    A0 = []
    PPP = []
    
    A = [1+2*i for i in range(5)]
    V = np.linspace(0.001,1,400)
    plt.figure()
    for aa in A:
        a0 = aa
        a,v = gradientmetode(a0/2, v0/2, 0.01, 100000)
        A0.append(v[-1])
        PPP.append(P(a[-1],v[-1]))
        
        PP = lambda t: P(aa,t)
        plt.plot(V,PP(V), label=f'$a^*=${aa}')
    
    plt.plot(A0,PPP,'o')
    plt.ylim(0,4); plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title(f'Snitkurver $P(a^*,v)$')
    
    plt.figure()
    for aa in A:
        plt.plot(V,dP_dv(aa,V), label=f'$a^*=${aa}')
    plt.plot(A0,[0 for i in range(len(A0))],'o')
    plt.ylim(-1,1); plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title(f'Snitkurver $P(a^*,v)$')

def gradientmetode2(a,v,h,N):
    """
    Anvender gradientmetoden.

    Parametre
    ----------
    a : Gaet paa a's vaerdi i maksimum
    v : Gaet paa v's vaerdi i maksimum
    h : Skridtlaengde
    N : Antal iterationer

    Returnerer
    -------
    A,V : punktfoelge
    """
    
    if a<=0 or a>a0 or v<=0 or v>v0: # Tjekker at gaettet er i den detaljeret region
        return print('not legal guess')
    
    A, V = [a], [v]
    for i in range(N):
        dPa, dPv = gradient(a,v) # Udregner gradienten
        length = np.sqrt(dPa**2+dPv**2)
        a_temp = a+h*dPa/length # Danner nyt punkt
        v_temp = v+h*dPv/length
        
        if a_temp <=0: # Overholder graenser for a
            a = 0.001
        elif a_temp > a0:
            a = a0
        else:
            a = a_temp
        
        if v_temp <=0: # Overholder graenser for v
            v = 0.001
        elif v_temp > v0:
            v = v0
        else:
            v = v_temp
        
        A.append(a); V.append(v)
        """
        if len(A) > 10: # Stopper hvis de seneste punkter er indenfor en hvis tolerance
            temp_dist = P(A[-1],V[-1])-P(A[-10],V[-10])
            if temp_dist < 0.000000001:
                break
        """
    return A,V

def k_test():
    global a0
    a0 = 1000
    global dT, Nu, f
    dTL = [(1+i)/np.pi for i in range(5)]
    NuL = [1+i for i in range(5)]
    fL = [(1+i)/np.pi for i in range(5)]
    
    Nu, f = 1,1/np.pi
    plt.figure()
    
    for l in dTL:
        dT = l
        a,v = gradientmetode(a0-200, v0/4, 0.01, 100)
        plt.plot([i for i in range(len(a))], P(np.array(a),np.array(v)),label=f'$k_1=${np.pi*dT}')
        plt.ylim(0,5.2); plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.title(f'Variation af $k_1$ med $k_2=${Nu} og $k_3=${np.pi*f}')
 
        
    dT, f = 1/np.pi,1/np.pi
    plt.figure()
    
    for l in NuL:
        Nu = l
        a,v = gradientmetode(a0-200, v0/4, 0.01, 100)
        plt.plot([i for i in range(len(a))], P(np.array(a),np.array(v)),label=f'$k_2=${Nu}')
        plt.ylim(0,5.2); plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.title(f'Variation af $k_2$ med $k_1=${np.pi*dT} og $k_3=${np.pi*f}')
        
    dT, Nu = 1/np.pi,1
    plt.figure()
    
    for l in fL:
        f = l
        a,v = gradientmetode(a0-200, v0/4, 0.01, 100)
        plt.plot([i for i in range(len(a))], P(np.array(a),np.array(v)),label=f'$k_3=${np.pi*f}')
        plt.ylim(0,1.2); plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.title(f'Variation af $k_3$ med $k_1=${np.pi*dT} og $k_2=${Nu}')
     

#k_test()

#snitkurver()
