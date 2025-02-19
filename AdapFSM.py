import numpy as np
import matplotlib.pyplot as plt 

# Construction of the Fisher Symmetric Measurements
def FSM( d, phase=1 ): #con phase=1, construye el povm negativo

    n = 2*d - 1
    psi_k = np.zeros( (d, n), dtype=complex )

    psi_0 = np.ones(d, dtype=complex) 
    psi_0[1:] = - phase* np.exp( 1j*np.pi/4 )
    psi_0 = psi_0 / np.sqrt( n )

    psi_k[:,0] = psi_0

    for k in range(1,d):
        psi = np.ones(d, dtype=complex) 
        psi[1:] = - phase*np.exp( 1j*np.pi/4 ) / ( np.sqrt(n) + 1 )
        psi[k] += phase*np.sqrt(0.5*n)
        psi = psi / np.sqrt(n)

        psi_k[:,2*k-1] = psi 

        psi = np.ones(d, dtype=complex) 
        psi[1:] = - phase*np.exp( 1j*np.pi/4 ) / ( np.sqrt(n) + 1 )
        psi[k] += phase*1j*np.sqrt(0.5*n)
        psi = psi / np.sqrt(n)

        psi_k[:,2*k] = psi 

    return psi_k 

#Elegir Estado Random
def RandomState( d ): 
    psi = np.random.randn( d ) + 1j*np.random.randn( d )
    psi = psi / np.linalg.norm( psi )
    return psi

#Medida Simulada
def SimMeas( state, measure, shots):

    probs = np.abs( measure.T.conj() @ state )**2

    result = np.random.multinomial(shots, probs , size=1)

    NewDistribution = result/shots

    return  NewDistribution[0]

#Cálculo de la fidelidad
def Fidelity( psi, phi ):  
    return np.abs(np.vdot( psi, phi ) )**2

##################
from scipy.optimize import least_squares

def complex2real( A ):
    return np.concatenate( ( np.real(A), np.imag(A) ), axis=0 )    

def real2compelex( A ):
    d  = len(A)//2 
    return A[:d] + 1j*A[d:] 

#Maximum Likelihood Estimation for pure states
def MLE_pure( d, probs_ex, 
                measures, 
                init_state=None ):  
    
    if init_state is None:
        init_state = RandomState( d )

    def fun( psi_real ): 
        psi = real2compelex( psi_real )
        psi = psi / np.linalg.norm( psi ) 
        probs_th = np.abs( measures.T.conj() @ psi ) ** 2
        return probs_th - probs_ex

    results = least_squares( fun, complex2real(init_state) )

    state_est = real2compelex( results.x )
    state_est = state_est / np.linalg.norm( state_est )
    
    return state_est


## Tomografía analítica usando mediciones Fihser Simétricas incluyendo procesamiento de los datos con MLE en cada estimación
def TomoAnaliticaMLE(state,ensamble, shots_1 , shots_2):
    #state : estado puro desconocido
    #ensamble : cantidad de copias del estado desconocido
    #shots_1: fracción del ensamble usado para medir los FSM + y -
    #shots_2: fracción del ensamble usado para medir el FSM con reconstrucción en MLE

    d = len(state)
    
    #PASO 0: medir base computacional y definir estado fiducial
    BaseComp = np.eye(d, dtype=complex)
    meas = SimMeas(state, BaseComp, shots=1)

    #Elegir estado fiducial como el que tuvo mayor probabilidad
    b0 = np.max(meas)
    fid = np.where(meas == b0, meas, 0)/b0
    index = np.where(meas == b0)[0][0]
    # cambio de base, para que el fiducial sea el estado cero
    Base = BaseComp
    Base[:,0] = fid
    Base, R = np.abs(np.linalg.qr( Base )) #Gram-Schidth

    #PASO 1: construir FSM, medirlos, encontrar coeficientes y fases, primera estimación
    fsm_plus = FSM( d , -1 )
    fsm_plus_1 = Base@fsm_plus
    measures_accumulated = fsm_plus_1

    fsm_minus = FSM( d )
    fsm_minus_1 = Base@fsm_minus
    measures_accumulated = np.concatenate([measures_accumulated, fsm_minus_1], axis=1)

    #Medidas simuladas
    probs_plus  = SimMeas(state, fsm_plus_1, shots_1*ensamble/2)
    probs_minus = SimMeas(state, fsm_minus_1, shots_1*ensamble/2)

    probs_accumulated = probs_plus
    probs_accumulated = np.concatenate([probs_accumulated, probs_minus], axis=0)

    #Coeficientes de los elementos de medición
    b0 = fsm_plus[0]
    bk = fsm_plus[1:].real
    ck = fsm_plus[1:].imag

    #Cálculo de coeficientes y fases
    Bk = np.sum( 0.5 * ( bk/b0 ) * ( probs_plus - probs_minus ), axis=1 ) 
    Ck = np.sum( 0.5 * ( ck/b0 ) * ( probs_plus - probs_minus ), axis=1 ) 

    angles = np.arctan2( Ck.real , Bk.real )

    #Ecuación para b0^2

    beta0 = np.zeros(2*d-1)

    for i in range(2*d-1):
        num = ( b0[i] )**2*np.sum( Bk**2 + Ck**2 ) - np.abs( np.sum( (bk[:,i]-1j*ck[:,i]) * (Bk + 1j*Ck) ) )**2
        den = 4*( b0[i] )**2 - 2*( probs_plus[i] + probs_minus[i] )

        beta0[i] = np.real(2*np.sqrt( num / den ))

    mean_beta0 = np.mean(beta0)

    betak = np.sqrt( Bk**2 + Ck**2 ) / mean_beta0

    state_0 = np.hstack( ( mean_beta0 , betak*np.exp( 1j*angles )))
    state_hat = Base @ ( state_0 / np.linalg.norm( state_0 ) )

    #PASO 2: utilizar MLE para refinar la estimación

    state_hat_est = MLE_pure( d, probs_accumulated, measures_accumulated, state_hat ) #agregar datos anteriores
    Fid_1 = Fidelity(state_hat_est, state)
    
    #PASO 3: crear FSM con estimación anterior, medir y reconstruir con MLE

    Base = BaseComp
    Base[:,0] = state_hat_est
    Base, R = np.linalg.qr( Base ) #Gram-Schmidth
    fsm = Base @ fsm_minus
    measures_accumulated_4 = np.concatenate([measures_accumulated,fsm], axis=1)
    probs_3 = SimMeas( state, fsm , shots_2*ensamble)
    probs_accumulated_4 = np.concatenate([probs_accumulated,probs_3], axis =0)
    state_est = MLE_pure( d, probs_accumulated_4, measures_accumulated_4, state_hat_est ) #agregar datos anteriores

    Fid_2 = Fidelity(state_est, state)
    
    return Fid_1 , Fid_2