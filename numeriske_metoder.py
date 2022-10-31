# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:52:23 2021

@author: Mikkel Hviid Thorn

Mange af funktionerne er vektoriserbarer, hvilket kan være brugbart i praktisk.

import:
    import numeriske_metoder as numm
    
Skrivemåde for simple funktioner;
    Hvis man ønsker en 'normal' matematisk funktion, er der en god kompakt skrivemåde
    kaldet lambda funktioner.
    f = lambda x,y: np.sin(x)*y + x**3*np.exp(3/y) + 4

    Dette givet dog kun mulighed for en simpel funktion og ikke detaljeret funktioner.
    
Indhold;
  - Polynomier (linje 34)
  - Nulpunktsalgoritme / iterationsformler (linje 205)
  - Differensformler / numerisk differentiation (linje 347)
  - Kvadraturregler / numerisk integration (linje 452)
  - Skridtformler / numerisk løsning af differentialligninger (linje 701)
  - Forskellige simulationer (differentialligningen) (linje 1007)
  
For at importere modulet
import sys
sys.path.append(r'C:\\Users\Mikkel Hviid Thorn\OneDrive\Documents\4 AAU - Matematik\2. semester\Computerstøttede beregninger\Modul 3-7')
"""

import numpy as np
import matplotlib.pyplot as plt





"""
Polynomier som approksimation
"""

def lagrange(x,x_values=None,y_values=None,f=None,N=None):
    """
    Danner interpolations polynomium på Lagrange form. Enten kan en funktion gives,
    hvor der så dannes ækvidistante punkter, ellers så gives (x,y) sæt.
    Bemærk at kun x_values/y_values eller f/N skal opgives, ikke begge!

    Parametre
    ----------
    x : liste
        Liste af x-værdier, hvor y-værdier ønskes approksimeret.
    x_values : liste
        Liste af x-værdier i de punkter, hvor polynomiet skal have interpolationsegenskab.
    y_values : liste
        Liste af y-værdier i de punkter, hvor polynomiet skal have interpolationsegenskab.
    f : funktion
        Funktion, hvor y-værdier ønskes approksimeret.
    N : naturligt tal
        Ordenen på polynomiet.

    Returnerer
    -------
    y : liste
        Liste af y-værdier, tilhørerne x-værdierne.
    """
    
    def l(k,x): # Lagrangepolynomier for det 'k'te punktsæt
        faktors = [(x-x_values[j])/(x_values[k]-x_values[j]) for j in range(len(x_values)) if j != k]
        return np.prod(faktors)
    
    if x_values != None and y_values != None: # Hvis (x,y) punkter er givet, hvor polynomiet ønskes at have interpolationsegenskab
        if type(x) == int or type(x) == float:
            p = [[y_values[k]*l(k,x) for k in range(len(x_values))]]
        else:
            p = [[y_values[k]*l(k,x[i]) for k in range(len(x_values))] for i in range(len(x))]
        
    elif f != None and N != None: # Hvis funktionen er givet og ordenen
        x_values = np.linspace(x[0],x[-1],N+1) # Danner ækvidistante punkter
        y_values = f(x_values) # Tilsvarende y-værdier
        
        if type(x) == int or type(x) == float:
            p = [[y_values[k]*l(k,x) for k in range(len(x_values))]]
        else:
            p = [[y_values[k]*l(k,x[i]) for k in range(len(x_values))] for i in range(len(x))]
        
    else: # Hvis inputtet ikke er fyldestgørende
        return print('Not right input!')
    
    return np.sum(p,axis=1) # Summer alle ledende for alle x-værdierne ('y' listen)

def taylor(f,x,a,N,h=0.1):
    """
    Danner Taylorpolynomium af orden 'N' i udviklingspunktet 'a'.

    Parametre
    ----------
    f : funktion
        Funktionen som Taylorapproksimeres.
    x : liste
        Liste af x-værdier, hvor y-værdier ønskes approksimeret.
    a : tal
        Udviklingspunktet, eller 'centrum' for Taylorapproksimationen.
    N : naturligt tal
        Ordenen på polynomiet.
    h : tal
        Skridtlængde til numerisk approksimation af den afledte. Her skal der
        både overvejes, hvor mange gange man skal udregne den afledte og
        hvordan skridtlængden vil multiplicerer.

    Returnerer
    -------
    y : liste
        Liste af y-værdier, tilhørerne x-værdierne.
    """
    
    def derivative(f,j,a): # Approksimere den 'j'te afledte evalueret i 'a'
        if j==1: # Hvis den kun skal differentieres numerisk en gang
            return df_sym_tre(f,a,h)
        else: # Hvis den skal differentieres flere gange returneres den differentieret funktion
            return derivative(lambda t: df_sym_tre(f,t,h),j-1,a)

    coef = [f(a)] # Listen med de afledte evalueret i 'a'
    for i in range(1,N+1):
        coef.append(derivative(f,i,a))
    
    def t(x): # Evaluere en 'x'-værdi for Taylorpolynomiet
        return np.sum([coef[i]/np.math.factorial(i)*(x-a)**i for i in range(N+1)])
        
    y = [t(x[i]) for i in range(len(x))]
    
    return y

def mikkelnomium(f,x,a,b,N,h=0.1):
    """
    mikkelnomium er lidt et joke navn for et polynomium jeg dannede i min SRP.
    Det er på en måde omvendt af Taylorpolynomier, da det anvender gentagne integraler
    istedet for afledte.
    
    En stor udfordring er at approksimere de gentagne integraler, hvilket kan være krævende.
    Dog ved at bruge Cauchys formel, kan dette opnås.
    
    Polynomiet fungerer meget godt indenfor [a,b], men divergerer hurtigt udenfor.

    Parametre
    ----------
    f : funktion
        Funktionen som approksimeres.
    x : liste
        Liste af x-værdier, hvor y-værdier ønskes approksimeret.
    a : tal
        Nedre grænse for approksimations intervallet.
    b : tal
        Øvre grænse for approksimations intervallet.
    N : naturligt tal
        Ordenen på polynomiet.
    h : tal
        Skridtlængde til numerisk approksimation af integralerne. Her skal der
        både overvejes, hvor mange integraler man skal udregne.
        
    Returnerer
    -------
    y : liste
        Liste af y-værdier, tilhørerne x-værdierne.
    """
    
    # h = (b-a)/(30*int((b-a))+1) Alternativ adaptiv valg af 'h'
    n = int((b-a)/h) # Antallet af indellinger til numerisk integration
    
    def integrate(f,j): # Numerisk 'j' gentagne integrale
        if j==1: # Det normale integrale
            return kvadratur_sammen(f,a,b,n,midtpunkt)
        else:
            g = lambda x: ((b-x)**(j-1))/np.math.factorial(j-1)*f(x)
            return kvadratur_sammen(g,a,b,n,midtpunkt) # Det gentagne integrale med Cauchys formel
        
            # Anvendelse af stamfunktioner og rekursion. For krævende og upræcis metode.
            #return integrate(lambda t: kvadratur_sammen(f,a,t,int((t-a)/h),midtpunkt),j-1,a,b)
            
            # Numerisk metode med teoretiske formler. Krævende, men meget upræcis og divergererne metode
            #k1 = 1/np.math.factorial(j-1)
            #l1 = [np.prod([n-i+k for k in range(1,j)]) for i in range(n)]
            #print(l1[0])
            #l2 = [f(a+h*i) for i in range(n)]
            #return np.sum([k1*l1[i]*l2[i]*h**j for i in range(n)])
    
    F_list = [f(b)] # Listen af de gentagne integraler evalueret i intervallet [a,b]
    for i in range(1,N+1):
        F_list.append(integrate(f,i))
    
    A = [[(b-a)**i for i in range(N+1)]] # En faktor/forholds matrice.
    for j in range(1,N+1):
        A.append([np.math.factorial(i)/np.math.factorial(j+i)*(b-a)**(i+j) for i in range(N+1)])
    
    # Den inverse 'A' matrix ganges med 'F_list' for at danne koefficient vektoren 'coef'
    coef = np.matmul(np.linalg.inv(np.array(A)),np.array(F_list))
    
    def m(x): # Evaluere en 'x'-værdi for polynomiet
        X0 = np.array([[(x-a)**i for i in range(N+1)]]) # Vektor med 'x'-faktorer
        return np.matmul(X0,coef) # Produkt mellem 'x'-faktorer og koefficienter

    y = [m(x[i]) for i in range(len(x))]

    return y





"""
Nulpunktsalgoritmer (bisektionsmetode og forskellige iterationsmetoder)
"""

def bisekt(f,a,b,eps):
    """
    bisekt anvender bisektionsmetoden til at finde et nulpunkt.

    Parametre
    ----------
    f : funktion
        En reel funktion med et nulpunkt mellem 'a' og 'b'.
    a : tal
        En nedre grænse for, hvor 'f' har et nulpunkt.
    b : tal
        En øvre grænse for, hvor 'f' har et nulpunkt.
    eps : fejlgrænse
        Den acceptable fejl fra nul.

    Returnerer
    -------
    (a+b)/2 : tal
        Approksimation af nulpunktet.
    i : heltal
        Antal iterationer anvendt.
    """
    
    i = 0
    if f(a)*f(b) <= 0: # Tjekker om funktion har et nulpunkt
        while abs(b-a) > eps: # Tjekker om approksimationen er god nok
            m = (a+b)/2
            if f(m)*f(a) <= 0: # Finder hvilken af 'a' og 'b', der skal erstattes med 'm'
                b = m
            else: 
                a = m
            i += 1 # En iteration er brugt
        return (a+b)/2, i
    return print('wrong a and b values')

def itera(g,x0,eps,imax):
    """
    itera bruger iterationsmetoden til at finde et nulpunkt.

    Parametre
    ----------
    g : funktion
        Funktion til at løse fikspunkt ligningen g(x)=x. Altså g(x)=f(x)+x.
    x0 : tal
        Startgættet for fikspunktet.
    eps : fejlgrænse
        Den acceptable fejl fra nul.
    imax : heltal
        Maksimalt antal iterationer tilladt, før funktionen stopper.

    Returnerer
    -------
    xn : tal
        Approksimation af nulpunktet.
    i : heltal
        Antal iterationer anvendt.
    """
    
    xn = g(x0); i = 1 # 'xn' er 'x1' i dette tilfælde og en iteration er gået
    while abs(g(xn)-xn) > eps: # Tjekker om approksimationen er god nok
        xn = g(xn); i += 1 # En iteration
        if i == imax: # Tjekker om det maksimale antal iterationer er nået. Bemærk at 'i'=1 ikke vil få den til at stoppe.
            return xn, i
    return g(xn), i

def newton(f,fm,x0,eps,imax):
    """
    newton bruger Newtons iterationsmetode til at finde et nulpunkt. 
    itera kan gøre det samme som newton.

    Parametre
    ----------
    f : funktion
        En reel funktion med et nulpunkt tæt på 'x1'.
    fm : funktion
        Den afledte funktion til 'f'.
    x0 : tal
        Startgættet for fikspunktet/for nulpunktet.
    eps : fejlgrænse
        Den acceptable fejl fra nul.
    imax : heltal
        Maksimalt antal iterationer tilladt, før funktionen stopper.

    Returnerer
    -------
    xn : tal
        Approksimation af nulpunktet.
    i : heltal
        Antal iterationer anvendt.
    """
    
    g = lambda x: x - f(x)/fm(x) # Definerer vores funktione til fikspunktligningen g(x)=x.
    xn = g(x0); i = 1 # 'xn' er 'x1' i dette tilfælde og en iteration er gået
    while abs(xn-g(xn)) > eps: # Tjekker om approksimationen er god nok
        xn = g(xn); i += 1 # En iteration
        if i == imax: # Tjekker om det maksimale antal iterationer er nået. Bemærk at 'i'=1 ikke vil få den til at stoppe.
            return xn, i
    return xn, i

def sekant(f,x1,x2,eps,imax):
    """
    sekant bruger en approksimation til Newtons iterationsmetode med en sekant. 
    itera kan ikke gøre det samme som sekant.

    Parametre
    ----------
    f : funktion
        En reel funktion med et nulpunkt tæt på 'x1' og 'x2'.
    x1 : tal
        Et tal tæt på nulpunktet.
    x2 : tal
        Et tal tæt på nulpunktet.
    eps : fejlgrænse
        Den acceptable fejl fra nul.
    imax : heltal
        Maksimalt antal iterationer tilladt, før funktionen stopper.

    Returnerer
    -------
    x3 : tal
        Approksimation af nulpunktet.
    i : heltal
        Antal iterationer anvendt.
    """
    
    g = lambda x,y: x - f(x)*(x-y)/(f(x)-f(y))
    x3 = g(x1,x2); i = 1
    while abs(x3-x2) > eps:
        x2, x1 = x3, x2
        x3 = g(x1,x2); i += 1
        if i == imax:
            return x3, i
    return x3, i





"""
Numeriske differentiation (differensformler)
Alle funktionerne bruger samme parametre og returnerer enten hældningen y' eller konkaviteten y''.


Parametre
    ----------
    f : funktion
        Funktionen som der ønskes numerisk differentieret.
    x : tal
        Tallet, hvor hældningen ønskes.
    h : tal
        Skridtlængden fra 'x', hvor approksimationen dannes.
        
Returnerer
    -------
    y': tal
        Approksimation af den afledte eller dobbelt afledte i 'x'.
        
        
Ordbog for funktionsnavne:
    df         - den afledte funktion y'
    dd         - den dobbelt afledte funktion y''
    fremad     - fremadgående metode
    tilbage    - tilbagegående metode
    sym        - symmetrisk metode
    to         - topunkts metode
    tre        - trepunkts metode
"""

def df_fremad_to(f,x,h):
    return (f(x+h)-f(x))/h

def df_tilbage_to(f,x,h):
    return (f(x)-f(x-h))/h

def df_sym_tre(f,x,h):
    return (f(x+h)-f(x-h))/(2*h)

def ddf_sym_tre(f,x,h):
    return (f(x+h)-2*f(x)+f(x-h))/(h**2)

def df_fremad_tre(f,x,h):
    return (-f(x+2*h)+4*f(x+h)+3*f(x))/(2*h)

def df_tilbage_tre(f,x,h):
    return (3*f(x)-4*f(x+h)+f(x+2*h))/(2*h)

def ddf_fremad_tre(f,x,h):
    return (f(x+2*h)-2*f(x+h)+f(x))/(h**2)

def ddf_tilbage_tre(f,x,h):
    return (f(x)-2*f(x-h)+f(x-2*h))/(h**2)


"""
Numerisk eksperiment med ordenen af metoderne
"""

def eksp_orden(f,df,numdf,x,N):
    """
    eksp_orden laver et numerisk eksperiment med en hvis differensformel.

    Parametre
    ----------
    f : funktion
        En reel funktion med et nulpunkt tæt på 'x1'.
    df : funktion
        Den afledte funktion til 'f'.
    numdf : funktion
        Den differensformel man ønsker at eksperimere med. f.eks. numdf=df_fremad_tre.
    x : tal
        Tallet, hvor hældningen ønskes.
    N : heltal
        Hvor mange iterationer eksperimentet er over. Forslag N=8-10.

    Returnerer
    -------
        Danner en tabel.
    """
    
    Error = [0.0 for i in range(N)] # Danner lister for fejlene og forholdet mellem fejlene
    Ratio = [0.0 for i in range(N)]
    
    Dexact = df(x)
    for k in range(N): # Udregner fejlene
        h = 2**(-k-2)
        Error[k] = abs(numdf(f,x,h) - Dexact)
    for k in [i+1 for i in range(N-1)]: # Udregner forholdet mellem fejlene
        Ratio[k] = Error[k-1]/Error[k]
    
    
    print('\nh=2**(-j-2), x0=%g\n' % x) # Danner tabel overskrift og konstruktion
    header_fmt='{0:>4} {1:^11} {2:^9}'
    print(header_fmt.format('j', 'Fejl', 'Forhold'))
    print(header_fmt.format('-'*4, '-'*11, '-'*9))
    
    for k in range(N):
        print('{0:>4g} {1:<6.5E} {2:<4.3E}'.format(k+2, Error[k], Ratio[k])) # Printer tabel
    return x





"""
Numeriske integration (kvadraturregler)
Alle funktionerne bruger samme parametre og returnerer approksimation af integralet.


Parametre
    ----------
    f : funktion
        Funktionen som der ønskes numerisk integreret.
    a : tal
        Den nedre grænse for intervallet, hvor funktionen ønskes integreres.
    b : tal
        Den øvre grænse for intervallet, hvor funktionen ønskes integreres.
        
Returnerer
    -------
    F(b)-F(a) : tal
        Approksimation af integralet af 'f' fra 'a' til 'b'.
"""

def midtpunkt(f,a,b):
    m = (b+a)/2
    return (b-a)*f(m)

def trapez(f,a,b):
    return (f(b)+f(a))*(b-a)/2

def simpson(f,a,b):
    m = (b+a)/2
    return (f(b)+4*f(m)+f(a))*(b-a)/6

def booles(f,a,b):
    x1, x2, x3 = (b+a)/4, (b+a)/2, 3*(b+a)/4
    return (7*f(b)+32*f(x1)+12*f(x2)+32*f(x3)+7*f(a))*(b-a)/90


"""
Sammensatte og adaptive kvadraturregler
"""

def kvadratur_sammen(f,a,b,n,kvadratur):
    """
    kvadratur_sammen splitter et interval i lige store delintervaller for 
    at opnå en bedre approksimation af integralet.
    
    Parametre
    ----------
    f : funktion
        Funktionen som der ønskes numerisk integreret.
    a : tal
        Den nedre grænse for intervallet, hvor funktionen ønskes integreres.
    b : tal
        Den øvre grænse for intervallet, hvor funktionen ønskes integreres.
    n : heltal
        Antallet af delintervaller, som [a,b] skal splittes til.
    kvadratur : funktion
        Den kvadraturregel man ønsker at bruge. f.eks. kvadratur=simpson.
        
    Returnerer
        -------
        F(b)-F(a) : tal
            Approksimation af integralet af 'f' fra 'a' til 'b'.
    """
    
    x = np.linspace(a,b,n+1) # Skaber ækvidistante x-værdier, som splitter [a,b] i delintervaller
    S = 0
    for i in range(n): # Bruger samme kvadraturregel for alle delintervaller og tager summen
        S += kvadratur(f,x[i],x[i+1])
    return S

def kvadratur_adap(f,a,b,tol,kvadratur,K=3/4):
    """
    kvadratur_adap prøver at splitte intervallet i smarte inddelinger.
    Hvis et delinterval har for stor antaget fejl, så splittes det i to lige store delintervaller.

    Parametre
    ----------
    f : funktion
        Funktionen som der ønskes numerisk integreret.
    a : tal
        Den nedre grænse for intervallet, hvor funktionen ønskes integreres.
    b : tal
        Den øvre grænse for intervallet, hvor funktionen ønskes integreres.
    tol : tal
        Den tolerence man ønsker at arbejde med. Bemærk at det ikke er nogen garanti, men et gæt.
    kvadratur : funktion
        Den kvadraturregel man ønsker at bruge. f.eks. kvadratur=simpson.
    K : tal
        Proposionalitet mellem kvadraturerne og fejlen. 3/4 er for midtpunkt og 3 er for Trapez.

    Returnerer
        -------
        F(b)-F(a) : tal
            Approksimation af integralet af 'f' fra 'a' til 'b'.
    """
    
    x = [a,b]; i = 0; S = 0 # Setup
    while i != len(x)-1: # Fortsætter indtil at alle delintervaller overholder tolerencen
        diff = kvadratur(f,x[i],x[i+1]) - kvadratur_sammen(f,x[i],x[i+1],2,kvadratur)
        if abs(diff) < K*tol*((x[i+1]-x[i])/(b-a)): # Tjekker om tolerencen overholdes for delintervallet [x[i],x[i+1]]
            S += kvadratur_sammen(f,x[i],x[i+1],2,kvadratur) # Tilføjer approksimation over intervallet.
            i += 1
        else:
            x.insert(i+1, (x[i+1]+x[i])/2) # Splitter delintervallet op i to.
    return S


"""
Udledt kvadraturregel fra kursusgang 7 opgave 3
"""

def kvadrat(f,h):
    """
    Parametre
    ----------
    f : funktion
        Funktionen som der ønskes numerisk integreret.
    h : tal
        Skridtlængde.

    Returnerer
    -------
    F(b)-F(a) : tal
        Approksimation af integralet af 'f' fra 'a' til 'b'.
    """
    
    a, b = 8/3*h, -4/3*h # Konstanter udledt i opgaven
    return a*f(h)+b*f(2*h)+a*f(3*h)

def kvadrat_sammen(f,h,n):
    """
    kvadrat_sammen er den sammensatte kvadratur af kvadraturreglen i funktionen 'kvadrat'.

    Parametre
    ----------
    f : funktion
        Funktionen som der ønskes numerisk integreret.
    h : tal
        Skridtlængde.
    n : heltal
        Antallet af delintervaller, som [a,b] skal splittes til.

    Returnerer
    -------
    F(b)-F(a) : tal
        Approksimation af integralet af 'f' fra 'a' til 'b'.
    """
    
    hi = h/n # ny skridtlængde til delintervallerne
    h_list = np.arange(0,4*h,4*hi) # Grænser for forskellige delintervaller
    a, b = 8/3*hi, -4/3*hi # Konstanter udledt i opgaven
    S = 0
    for i in range(n): # Bruger samme kvadraturregel for alle delintervaller og tager summen
        S += a*f(hi+h_list[i])+b*f(2*hi+h_list[i])+a*f(3*hi+h_list[i])
    return S

def kvadrat_alt(f,a,b):
    """
    kvadrat_alt er en alternativ formulering af kvadraturreglen 'kvadrat'.
    Den har samme parametre og fungerer ens med de andre basiske kvadraturregler.
    Den kan bruges i 'kvadratur_sammen' og 'kvadratur_adap'.
    """
    
    h = (b-a)/4
    k1, k2 = 8/3*h, -4/3*h
    return k1*f(h+a)+k2*f(2*h+a)+k1*f(3*h+a)
        

"""
Romberg integration
"""    

def romberg(f,a,b,M=10):
    """
    Funktionen 'romberg' danner flere numeriske approksimationer af integralet af 'f' i [a,b] 
    efter Romberg integration reglen.

    Parametre
    ----------
    f : funktion
        Funktionen som der ønskes numerisk integreret.
    a : tal
        Den nedre grænse for intervallet, hvor funktionen ønskes integreres.
    b : tal
        Den øvre grænse for intervallet, hvor funktionen ønskes integreres.
    M : heltal
        Maksimal Romberg integration der dannes. 
        Den bedste approksimation må antages at blive R_{M,M} i dette tilfælde.

    Returnerer
    -------
    R : matrix
        Returnerer en matrix i hvor indgangende svarer til Romberg integrationen R_{i,j}.
        Hvis i<j, så er indgangen nul.
    """
    
    N = 1
    R = np.zeros((M, M)) # Danner 'R' matricen som returneres i enden
    R[0, 0] = trapez(f,a,b) # Udregner den første approksimation med Trapez reglen
    for L in range(1,M): # Rækker i 'R'
        I = kvadratur_sammen(f,a,b,N,midtpunkt)
        R[L, 0] = (R[L - 1, 0] + I) / 2 # Først udregnes første indgang i en række
        for k in range(L): # Udfyldelse af rækken 'L'
            R[L, k + 1] = ((4**(k + 1) * R[L, k] - R[L - 1, k]) /
                           (4**(k + 1) - 1))
        N *= 2
    return R

def romberg_adap(f,a,b,tol=0.01):
    """
    Funktionen 'romberg_adap' stopper først, når det er antaget, at en tolerence 
    er opnået for fejlen mellem integralet og R_{M,M}.

    Parametre
    ----------
    f : funktion
        Funktionen som der ønskes numerisk integreret.
    a : tal
        Den nedre grænse for intervallet, hvor funktionen ønskes integreres.
    b : tal
        Den øvre grænse for intervallet, hvor funktionen ønskes integreres.
    tol : tal
        Den tolerence man ønsker at arbejde med. Bemærk at det ikke er nogen garanti, men et gæt.

    Returnerer
    -------
    R : matrix
        Returnerer en matrix i hvor indgangende svarer til Romberg integrationen R_{i,j}.
        Hvis i<j, eksisterer indgangen ikke.
    """
    
    N = 1; L = 0
    R = [[0]] # Danner 'R' matricen som returneres i enden
    R[L][0] = trapez(f,a,b) # Udregner den første approksimation med Trapez reglen
    while True: # Rækker i 'R'
        L += 1
        I = kvadratur_sammen(f,a,b,N,midtpunkt)
        R.append([0 for i in range(L+1)]) # Tilføjer resten af rækken
        R[L][0] = (R[L - 1][0] + I) / 2 # Først udregnes første indgang i en række
        for k in range(L): # Udfyldelse af rækken 'L'
            R[L][k + 1] = ((4**(k + 1) * R[L][k] - R[L - 1][k]) /
                           (4**(k + 1) - 1))
        N *= 2
        if abs(R[L][L]-R[L-1][L-1]) < tol: # Tjekker om tolerencen overholdes
            break
    return R





"""
Numerisk løsning af differentialligninger (Skridtformler)


Runge-Kutta anden-ordens metoder med bestemte alpha:
    alpha = 1/2     ->  rettet euler / eksplicit midtpunkt
    alpha = 1       ->  modificeret euler / eksplicit trapez
    alpha = 2/3     ->  Heuns metode
    

Hvis et system af differentialligninger ønskes at blive løst, 
så skal f være en vektor funktion, som tager et tal 'x' og en vektor 'y' ind. 
Se kursusgang 9.
Derudover skal 'ya' være en vektor med begyndelsesværdierne.
"""

def euler(f,a,b,ya,h):
    """
    euler anvender Eulers metode til at approksimere en løsning til en differentialligning i [a,b].

    Parametre
    ----------
    f : funktion
        Funktionen svarende til hældningsudtrykket f(t,y), altså f(t,y)=y'(t).
    a : tal
        Startværdien af intervallet.
    b : tal
        Slutværdien af intervallet.
    ya : tal
        Den kendte værdi, så y(a)=ya. Det er begyndelsesværdien til IVP.
    h : tal
        Skridtlængden for approksimationen.

    Returnerer
    -------
    p : liste
        'p' er en liste med to lister. Den første liste p[0] indeholder alle x-værdier.
        Den anden liste p[1] indeholder alle y-værdier.
    """
    
    p = [[a],[ya]] # Begyndelsesværdien
    while True: # Approksimationerne af de følgende punkter
        p[1].append(p[1][-1]+h*f(p[0][-1],p[1][-1]))
        p[0].append(p[0][-1]+h)
        if p[0][-1] >= b-0.00001: # Tjekker om intervallet [a,b] er dækket
            break
    return p

def rungekutta_anden(f,a,b,ya,h,alpha=1):
    """
    rungekutta_anden anvender Runge-Kutta anden ordens metoder til at approksimere 
    en løsning til en differentialligning i [a,b].

    Parametre
    ----------
    f : funktion
        Funktionen svarende til hældningsudtrykket f(t,y), altså f(t,y)=y'(t).
    a : tal
        Startværdien af intervallet.
    b : tal
        Slutværdien af intervallet.
    ya : tal
        Den kendte værdi, så y(a)=ya. Det er begyndelsesværdien til IVP.
    h : tal
        Skridtlængden for approksimationen.
    alpha : tal
        Værdi for skridtet til mellem skridtet i metoden.

    Returnerer
    -------
    p : liste
        'p' er en liste med to lister. Den første liste p[0] indeholder alle x-værdier.
        Den anden liste p[1] indeholder alle y-værdier.
    """
    
    p = [[a],[ya]] # Begyndelsesværdien
    while True: # Approksimationerne af de følgende punkter
        k1 = f(p[0][-1],p[1][-1])
        k2 = f(p[0][-1]+alpha*h,p[1][-1]+alpha*h*k1)
        p[1].append(p[1][-1]+h*(k1*(1-1/(2*alpha))+k2/(2*alpha)))
        p[0].append(p[0][-1]+h)
        if p[0][-1] >= b-0.00001: # Tjekker om intervallet [a,b] er dækket
            break
    return p

def eksplicit_simpson(f,a,b,ya,h):
    """
    eksplicit_simpson anvender en form for Simpsons kvadraturregel til at approksimere 
    en løsning til en differentialligning i [a,b].

    Parametre
    ----------
    f : funktion
        Funktionen svarende til hældningsudtrykket f(t,y), altså f(t,y)=y'(t).
    a : tal
        Startværdien af intervallet.
    b : tal
        Slutværdien af intervallet.
    ya : tal
        Den kendte værdi, så y(a)=ya. Det er begyndelsesværdien til IVP.
    h : tal
        Skridtlængden for approksimationen.

    Returnerer
    -------
    p : liste
        'p' er en liste med to lister. Den første liste p[0] indeholder alle x-værdier.
        Den anden liste p[1] indeholder alle y-værdier.
    """
    
    p = [[a],[ya]] # Begyndelsesværdien
    while True: # Approksimationerne af de følgende punkter
        k1 = f(p[0][-1],p[1][-1])
        k2 = f(p[0][-1]+h/2,p[1][-1]+h/2*k1)
        k3 = f(p[0][-1]+h,p[1][-1]-h*k1+2*h*k2)
        p[1].append(p[1][-1]+h/6*k1+2*h/3*k2+h/6*k3)
        p[0].append(p[0][-1]+h)
        if p[0][-1] >= b-0.00001: # Tjekker om intervallet [a,b] er dækket
            break
    return p





"""
De følgende numeriske løsningsmetoder til differentialligninger (Skridtformler)
kan ikke anvendes til et system.
"""

def implicit_euler(f,a,b,ya,h):
    """
    implicit_euler anvender Eulers metode bagvendt til at approksimere 
    en løsning til en differentialligning i [a,b].

    Parametre
    ----------
    f : funktion
        Funktionen svarende til hældningsudtrykket f(t,y), altså f(t,y)=y'(t).
    a : tal
        Startværdien af intervallet.
    b : tal
        Slutværdien af intervallet.
    ya : tal
        Den kendte værdi, så y(a)=ya. Det er begyndelsesværdien til IVP.
    h : tal
        Skridtlængden for approksimationen.

    Returnerer
    -------
    p : liste
        'p' er en liste med to lister. Den første liste p[0] indeholder alle x-værdier.
        Den anden liste p[1] indeholder alle y-værdier.
    """
    
    eps = 1.0e-10; imax = 100 # Setup
    
    p = [[a],[ya]] # Begyndelsesværdien
    while True: # Approksimationerne af de følgende punkter
        numerator = lambda z: p[1][-1]+ h*f(p[0][-1]+h,z) -z
        denominator = lambda z: df_sym_tre(numerator, z, h)
        y, i = newton(numerator, denominator, p[1][-1], eps, imax)
        p[1].append(y)
        p[0].append(p[0][-1]+h)
        if p[0][-1] >= b-0.00001: # Tjekker om intervallet [a,b] er dækket
            break
    return p

def implicit_trapez(f,a,b,ya,h):
    """
    implicit_trapez anvender Trapez kvadraturreglen til at approksimere 
    en løsning til en differentialligning i [a,b]. Den kaldes implicit, da y_{i+1}
    ikke approksimeres to gange.

    Parametre
    ----------
    f : funktion
        Funktionen svarende til hældningsudtrykket f(t,y), altså f(t,y)=y'(t).
    a : tal
        Startværdien af intervallet.
    b : tal
        Slutværdien af intervallet.
    ya : tal
        Den kendte værdi, så y(a)=ya. Det er begyndelsesværdien til IVP.
    h : tal
        Skridtlængden for approksimationen.

    Returnerer
    -------
    p : liste
        'p' er en liste med to lister. Den første liste p[0] indeholder alle x-værdier.
        Den anden liste p[1] indeholder alle y-værdier.
    """
    
    eps = 1.0e-10; imax = 100 # Setup
    
    p = [[a],[ya]] # Begyndelsesværdien
    while True: # Approksimationerne af de følgende punkter
        numerator = lambda z: p[1][-1]+ h*(f(p[0][-1],p[1][-1])+f(p[0][-1]+h,z))/2 -z
        denominator = lambda z: df_sym_tre(numerator, z, h)
        y, i = newton(numerator, denominator, p[1][-1], eps, imax)
        p[1].append(y)
        p[0].append(p[0][-1]+h)
        if p[0][-1] >= b-0.00001: # Tjekker om intervallet [a,b] er dækket
            break
    return p

def adaptiv_rk12(f,a,b,ya,tol=0.1):
    """
    adaptiv_rk12 anvender Runge-Kutta første ordens og anden ordens metoder
    til at approksimere en løsning til en differentialligning i [a,b] adaptivt.
    RK1 er valgt til Eulers metode og RK2 er valgt til eksplicit Trapez metoden.

    Parametre
    ----------
    f : funktion
        Funktionen svarende til hældningsudtrykket f(t,y), altså f(t,y)=y'(t).
    a : tal
        Startværdien af intervallet.
    b : tal
        Slutværdien af intervallet.
    ya : tal
        Den kendte værdi, så y(a)=ya. Det er begyndelsesværdien til IVP.

    Returnerer
    -------
    p : liste
        'p' er en liste med to lister. Den første liste p[0] indeholder alle x-værdier.
        Den anden liste p[1] indeholder alle y-værdier.
    """
    
    h = 0.5; i = 0 # Setup
    
    p = [[a],[ya]] # Begyndelsesværdien
    while True: # Approksimationerne af de følgende punkter
        y_euler = p[1][-1]+h*f(p[0][-1],p[1][-1])
        y_trapez = p[1][-1]+h*(f(p[0][-1],p[1][-1])+f(p[0][-1]+h,y_euler))/2
        i += 1
        if np.abs(y_euler-y_trapez) < tol:
            p[1].append(y_trapez)
            p[0].append(p[0][-1]+h)
            i = 0
            h = 0.8*np.sqrt(tol*np.abs(p[1][-1])/np.abs(y_euler-y_trapez))*h
        elif i == 1:
            h = 0.8*np.sqrt(tol*np.abs(p[1][-1])/np.abs(y_euler-y_trapez))*h
        else:
            h = h/2
        
        if p[0][-1] >= b-0.00001: # Tjekker om intervallet [a,b] er dækket
            break
    return p

def adaptiv_rk23(f,a,b,ya,tol=0.1):
    """
    adaptiv_rk12 anvender Runge-Kutta anden ordens og tredje ordens metoder
    til at approksimere en løsning til en differentialligning i [a,b] adaptivt.
    RK2 er valgt til eksplicit Trapez metoden og RK3 er valgt til eksplicit Simpsons metoden.

    Parametre
    ----------
    f : funktion
        Funktionen svarende til hældningsudtrykket f(t,y), altså f(t,y)=y'(t).
    a : tal
        Startværdien af intervallet.
    b : tal
        Slutværdien af intervallet.
    ya : tal
        Den kendte værdi, så y(a)=ya. Det er begyndelsesværdien til IVP.

    Returnerer
    -------
    p : liste
        'p' er en liste med to lister. Den første liste p[0] indeholder alle x-værdier.
        Den anden liste p[1] indeholder alle y-værdier.
    """
    
    h = 0.5; i = 0 # Setup
    
    p = [[a],[ya]] # Begyndelsesværdien
    while True: # Approksimationerne af de følgende punkter
        k1 = f(p[0][-1],p[1][-1])
        k2 = f(p[0][-1]+h/2,p[1][-1]+h/2*k1)
        k3 = f(p[0][-1]+h,p[1][-1]-h*k1+2*h*k2)
        
        y_trapez = p[1][-1]+h*(k1+k3)/2
        y_simpson = p[1][-1]+h/6*k1+2*h/3*k2+h/6*k3
        i += 1
        
        if np.abs(y_trapez-y_simpson) < tol:
            p[1].append(y_simpson)
            p[0].append(p[0][-1]+h)
            i = 0
            h = 0.8*np.sqrt(tol*np.abs(p[1][-1])/np.abs(y_simpson-y_trapez))*h
        elif i == 1:
            h = 0.8*np.sqrt(tol*np.abs(p[1][-1])/np.abs(y_simpson-y_trapez))*h
        else:
            h = h/2
        
        if p[0][-1] >= b-0.00001: # Tjekker om intervallet [a,b] er dækket
            break
    return p





"""
Tre legeme problemet

Der er tre legemer a, b og c
- 'm' er listen over masser: 
    m = [Am, Bm, Cm]
    
- 'g' er en tyngdekonstant, ofte sat til 1: 
    g = 1
    
- 'z0' er en vektor med begyndelsesværdierne, i en ordnede række følge. 
  Den indholder både positioner 'x' og 'y' og hastighed 'v':
    z0 = np.array([Ax, Avx, Ay, Avy,   Bx, Bvx, By, Bvy,   Cx, Cvx, Cy, Cvy,])
    
- 'tmax' er den maksimale tid simulation skal kører over:
    tmax = integer
    
- 'dt' er tidsskridtet:
    dt = 0.01 (0.1 eller mindre)
"""

def trelegeme(m,z0,tmax,dt,g=1):
    """
    Løser og illustrerer tre legeme problemet for tre legemer A, B og C.

    Parametre
    ----------
    m : liste
        m = [Am, Bm, Cm]
        'm' er listen over masser.
    g : tal
        Gravitationskonstant.
        g = 1
    z0 : vektor
        'z0' er en vektor med begyndelsesværdierne, i en ordnede række følge. 
        Den indholder både positioner 'x' og 'y' og hastighed 'v'.
        z0 = np.array([Ax, Avx, Ay, Avy,   Bx, Bvx, By, Bvy,   Cx, Cvx, Cy, Cvy,])
    tmax : tal
        'tmax' er den maksimale tid simulation skal kører over.
    dt : tal
        'dt' er tidsskridtet. Vælg højst 0.1.

    Returnerer
    -------
        Et plot over tre legeme systemet.
    """
    
    # Definer påvirkningen af accelerationen på et legeme fra et andet legeme i hver dimension.
    # Lad L1 være påvirket af L2. 'i' er indeks for L1's position, og 'j' er indeks for L2's position.
    # 'k' er L2's masse og 'z' er vektoren med hastigheder og positioner for legemerne.
    gx = lambda i,j,k,z: g*m[k]*(z[j]-z[i]) / (((z[j]-z[i])**2+(z[j+2]-z[i+2])**2)**(3/2))
    gy = lambda i,j,k,z: g*m[k]*(z[j]-z[i]) / (((z[j]-z[i])**2+(z[j-2]-z[i-2])**2)**(3/2))
    
    # Definer hældningsfelt funktionen. Den tager 't' tiden og 'z' vektoren med hastigheder og positioner for legemerne.
    # Funktionen giver en vektor, som er hældningerne for de forskellige værdier i 'z' til tiden 't'.
    f = lambda t, z: np.array([z[1], gx(0,4,1,z)+gx(0,8,2,z), z[3], gy(2,6,1,z)+gy(2,10,2,z),
                               z[5], gx(4,0,0,z)+gx(4,8,2,z), z[7], gy(6,2,0,z)+gy(6,10,2,z),
                               z[9], gx(8,0,0,z)+gx(8,4,1,z), z[11], gy(10,2,0,z)+gy(10,6,1,z)])
    
    # Runge-Kutta anden ordens metode (modificeret euler / eksplicit trapez) anvendes til numerisk løsning.
    res = rungekutta_anden(f,0,tmax,z0,dt,1)
    
    # Matricen 'pos' har rækker, der svarer til en position eller hastighed for et legeme.
    # Matricens søjler er 'z' vektoren til et bestemt tidspunkt.
    pos = np.array(res[1]).transpose()
    
    # Tager rækkevektorerne fra 'pos', som svarer til positioner.
    P1 = [pos[0] , pos[2]]
    P2 = [pos[4] , pos[6]]
    P3 = [pos[8] , pos[10]]
    
    # Alternativ løsning til anvendelsen af 'pos'.
    #P1 = [[res[1][k][0] for k in range(0,int(tmax/dt+1))] , [res[1][k][2] for k in range(0,int(tmax/dt+1))]]
    #P2 = [[res[1][k][4] for k in range(0,int(tmax/dt+1))] , [res[1][k][6] for k in range(0,int(tmax/dt+1))]]
    #P3 = [[res[1][k][8] for k in range(0,int(tmax/dt+1))] , [res[1][k][10] for k in range(0,int(tmax/dt+1))]]
    
    # Plotter de tre legemers positioner over tiden [0,tmax].
    # Plottes med stiplede linjer.
    plt.figure()
    plt.plot(P1[0], P1[1], '--', color='firebrick')
    plt.plot(P2[0], P2[1], '--', color='navy')
    plt.plot(P3[0], P3[1], '--', color='forestgreen')
    
    # Plotter de tre legemers sidste udregnet position, positionen til tiden 'tmax'.
    plt.plot(P1[0][-1], P1[1][-1], 'o', color='firebrick')
    plt.plot(P2[0][-1], P2[1][-1], 'o', color='navy')
    plt.plot(P3[0][-1], P3[1][-1], 'o', color='forestgreen')


"""
Tre legeme problemer med startværdier

# a (Sauer 6.3.3 opg 16)
# Systemet består af tre lige store legemer, der danner en stabil ottetals bane.

tmax1, dt1 = 3, 0.001
m1, g1 = [1, 1, 1], 1
z1 = np.array([-0.97,-0.466, 0.243,-0.433,   0.97,-0.466, -0.243,-0.433,   0,2*0.466, 0,2*0.433])



# b (Sauer 6.3.3 opg 12)
# Systemet består af et stort legeme og to mindre, hvor de to mindre legemer har stabile baner omkring det stillestående større legeme.

tmax2, dt2 = 50, 0.001
m2, g2 = [0.03, 0.3, 0.03], 1
z2 = np.array([2,0.2, 2,-0.2,   0,0, 0,0,   -2,-0.2, -2,0.2])



# (Sauer 6.3.3 opg 14)
# Systemet består af et stort legeme, et mindre og et meget lille legeme. Det store legeme og det mindre er stabilt, med det mindre havende
# en bane omkring det store. Det mindste legeme ligger længere fra begge, og har ikke en særlig stabil bane.

tmax3, dt3 = 500, 0.01
m3, g3 = [0.05, 1, 0.005], 1
z3 = np.array([0,0.6, 2,0.05,   0,-0.03, 0,0,   4,0, 3,-0.5])



# (Sauer 6.3.3 opg 15)
# Systemet består af et næsten stillestående større legeme og et mindre legeme, med en stabil bane omkring det større legeme. Det sidste
# legeme har elliptiske baner omkring det større legeme, hvor banen roterer rundt om det større legeme. 

tmax4, dt4 = 100, 0.001
m4, g4 = [0.05, 1, 10**(-5)], 1
z4 = np.array([0,0.6, 2,0,   0,-0.03, 0,0,   4,-0.2, 3,0])
"""





"""
SIR-model

SIR-model omhandler tre koblede differentialligninger og viser en epidemi udvikle sig.
"""

def SIR(N=1000,b=1,gamma=0.33,dt=0.01,tmax=40):
    """
    Danner graferne over en SIR-model med Eulers metode.

    Parametre
    ----------
    N : heltal
        Din population. Standard er 1000.
    b : tal
        Kontakttallet, som bestemmer hvor voldsom smitten er. Standard er 1.
    gamma : tal
        Helbredsraten, som bestemmer hvor hurtigt folk bliver 'helbredt'. 
        I modellen står det både for døde og helbredte. Standard er 0.33.
    dt : tal
        Tidsskridt. Standard er 1000.
    tmax : tal
        Sluttidspunktet.

    Returnerer
    -------
    S : punktmængde
        Grafen over de modtagelige (susceptible).
    I : punktmængde
        Grafen over de syge (inficerede).
    R : punktmængde
        Grafen over de immune og døde (removed).
    """
    
    a = b/N
    
    t = [0]
    S = [N-1]
    I = [1]
    R = [0]
    
    while t[-1] < tmax:
        dS = -a*S[-1]*I[-1]
        dI = a*S[-1]*I[-1]-gamma*I[-1]
        dR = gamma*I[-1]
        
        t.append(t[-1]+dt)
        S.append(dS*dt+S[-1])
        I.append(dI*dt+I[-1])
        R.append(dR*dt+R[-1])
        
    
    plt.style.use('seaborn')
    
    plt.figure()
    plt.plot(t, S, color='firebrick', label='S')
    plt.plot(t, I, color='navy', label='I')
    plt.plot(t, R, color='forestgreen', label='R')
    
    TitleString = 'SIR-model med dt=%g, N=%g, a=%g og gamma=%g' % (dt,N,a,gamma)
    plt.title(TitleString)
    plt.legend(loc = 'right')
    plt.xlabel('Tid'); plt.ylabel('Individer')





"""
Elatisk bold

Funktion til løsning af delopgave 2 i E-OPG2
"""

def elastisk_bold(r0,gamma=0.5,n_koll=1,tol=0.01,h0=0.5):
    """
    Simulation af en elatisk bold, der søger mod en gravitationsmidte.
    
    Modellen kommer fra en anden ordens differentialligning, som løses med
    Runge-Kutta par af anden og tredje orden, henholdsvis eksplicit Trapez og
    eksplicit Simpson.

    Parametre
    ----------
    r0 : tal
        Startafstanden mellem den elastiske bold og gravitationsmidten.
    gamma : tal
        gamma er det procentmæssige tab i hastighed ved kollision med gravitationsmidten.
    n_koll : heltal
        Antal gange bolden skal kollidere med gravitationsmidten.
    tol : tal
        Tolerencen for den adaptive metode.
    h0 : tal
        Standardskridtlængden.

    Returnerer
    -------
        Et plot over den elatiske bold.
    """
    
    h = h0; r_min = 0.1; kollisioner = 0 # Setup
    
    f = lambda y: np.array([y[1],-2/(y[0]**2)]) # Hældningsfunktionen
    
    p = [[0],[np.array([r0,0])]] # Begyndelsesværdien
    while True: # Begynder approksimation med eksplicit Trapez og Simpson
        k1 = f(p[1][-1])
        tilde_y_halv = p[1][-1]+h/2*k1
        k2 = f(tilde_y_halv)
        tilde_y = p[1][-1]-h*k1+2*h*k2
        k3 = f(tilde_y)
        
        e = np.abs(h/3*(2*k3-k1-k2))[0] # Fejlen/forskellen for 'r(t)', kan også vælge np.linalg.norm
        
        if e < tol: # Tjekker fejltolerencen
            y_simpson = p[1][-1]+h/6*k1+2*h/3*k2+h/6*k3 # Udregner eksplicit Simpsons metode
            
            if y_simpson[0] < r_min: # Tjekker for kollision
                kollisioner += 1
                
                # Danner et anden ordens interpoleret polynomium for 'r(t)-r_min'
                t_values = [p[0][-1], p[0][-1]+h/2, p[0][-1]+h]
                y_values = np.array([p[1][-1], tilde_y_halv, y_simpson]).transpose()
                poly = lambda t: lagrange(t,t_values,list(y_values[0]))[0] - r_min

                # Finder nulpunkt for det interpoleret polynomium
                t_hat = bisekt(poly, t_values[0], t_values[2], 0.0001)[0]
                
                # Finder hældningen i nulpunktet med et interpoleret polynomium
                diff_y = lagrange(t_hat,t_values,list(y_values[1]))
                
                # Opsamler og opdaterer punkter
                y = np.array([poly(t_hat) + r_min, -gamma*diff_y[0]])
                p[1].append(y)
                p[0].append(t_hat)

                if kollisioner == n_koll: # Tjekker om alle kollisioner er dannet
                    break
            
            else: # Tilføjer værdier uden kollision
                p[1].append(y_simpson)
                p[0].append(p[0][-1]+h)
            
            # Sætter skridtlængden til standardskridtlængden
            h = h0
            
        else: # Mindsker skridtlængden, hvis tolerencen ikke er overholdt
            h = 2*h/3

    y = np.array(p[1]).transpose() # Finder approksimation til 'r(t)'
    
    plt.figure() # Danner plot over simulationen
    plt.plot(p[0], y[0], '-', color='firebrick')
    plt.plot(p[0], y[0], '+', color='navy')
    TitleString = 'Elastisk bold med r0=%g, gamma=%g, hop=%g, tol=%g og h0=%g' % (r0,gamma,n_koll,tol,h0)
    plt.title(TitleString)
    plt.xlabel('Tid (t)'); plt.ylabel('Afstand (r)')




def exp():
    """
    Løser problemet 
    f(x,y)=y
    y(0)=1
    """
    plt.style.use('seaborn')
    x = [-5+0.01*i for i in range(1001)]; y = [1]; dx = 0.01
    for i in range(500):
        y = [y[0]-dx*y[0]] + y + [y[-1]+dx*y[-1]]
    print(y[600])
    plt.figure()
    plt.plot(x,y)

def sin_cos():
    """
    Løser problemet 
    f(x,y1,y2)=[y2,-y1]
    y(0)=[0,1]
    """
    plt.style.use('seaborn')
    x = [-5+0.01*i for i in range(1001)]; y1 = [0]; y2 = [1]; dx = 0.01
    for i in range(500):
        y1 = [y1[0]-dx*y2[0]] + y1 + [y1[-1]+dx*y2[-1]]
        y2 = [y2[0]+dx*y1[0]] + y2 + [y2[-1]-dx*y1[-1]]
    for i in range(500):
        if y1[501+i] < 0:
            a = (y1[501+i]-y1[500+i])/(x[501+i]-x[500+i])
            b = y1[501+i]-a*x[501+i]
            print(-b/a)
            break
    plt.figure()
    plt.plot(x,y1)
    plt.plot(x,y2)