import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit
from time import time, sleep
from sys import argv

def get_Globals():
    global NMOL, NEL, NPOL, INIT_BASIS, INIT_STATE
    NMOL       = int(argv[1]) # Number of molecules
    NEL        = 2
    NPOL       = 1 + NMOL*(NEL-1) + 1
    INIT_BASIS = "POL" # "adFock", "POL"
    INIT_STATE = -1

    global TIME, NSTEPS, dtN, MASS, NTRAJ, BATCH_SIZE
    NTRAJ  = 100
    NTIME  = 1 # ps
    dtN    = 0.5*41.341 # fs to a.u.
    NSTEPS = int(NTIME * 1000 * 41.341 / dtN) + 1 # 2_000
    TIME   = np.arange( 0, NSTEPS*dtN, dtN )
    MASS   = 1836.0

    global A0, WC
    A0 = 0.05
    WC = 0.085 # 0.085 is resonance at FC point

    # Langevin Part (Fluctuation-Dissipation Verlet Propagation)
    global L_COEFF, kT, a, b
    L_COEFF         = 1.0 # Friction Coefficient (i.e., How fast to get to kT temperature ?)
    #kT              = 300 * (0.025 / 300 / 27.2114) # Temperature in Hartree
    kT              = 200 * (0.025 / 300 / 27.2114) # Temperature in Hartree

    # Get memory size in GB of NMOL,NPOL,NPOL ndarray
    global MEMORY_SIZE
    MEMORY_SIZE = NMOL * NPOL * NPOL * 8 / 1024**3
    print( "Memory size of Force matrix (GB): %1.3f GB" % MEMORY_SIZE )
    if ( MEMORY_SIZE > 2 ):
        print( "Force matrix is too big for head node.\nI hope this is submitted to the cluster.\n\tIf not, kill this job." )
        sleep(10)
        #exit()

    global DATA_DIR
    DATA_DIR = "PLOTS_DATA_NTRAJ_%d_NMOL_%d/" % (NTRAJ, NMOL)
    try: os.mkdir(DATA_DIR)
    except FileExistsError: pass

def interpolate_Hel():

    R_D     = -9.5
    R_A     =  9.5
    WIDTH_D =  3.1
    WIDTH_A =  4.0
    WIDTH_H =  5.0
    r_min   = -30
    r_max   =  30
    Nr      =  400 # Choose even number...
    rgrid   = np.linspace(r_min, r_max, Nr)
    dr      = rgrid[1] - rgrid[0]
    
    from scipy.special import erf
    def get_V( R ): # Get the potential energy matrix for electron
        V     = np.zeros( Nr )
        V[:] -= erf( np.abs(rgrid - R_A) / WIDTH_A ) / np.abs(rgrid - R_A) # Electron-Acceptor Coulomb Attraction
        V[:] -= erf( np.abs(rgrid - R_D) / WIDTH_D ) / np.abs(rgrid - R_D) # Electron-Donor    Coulomb Attraction
        V[:] -= erf( np.abs(rgrid - R  ) / WIDTH_H ) / np.abs(rgrid - R  ) # Electron-Proton   Coulomb Attraction
        V[:] += 1 / np.abs( R   - R_A ) # Proton-Acceptor Coulomb Repulsion
        V[:] += 1 / np.abs( R   - R_D ) # Proton-Donor    Coulomb Repulsion
        V[:] += 1 / np.abs( R_D - R_A ) # Acceptor-Donor  Coulomb Repulsion
        return np.diag(V)

    # Get the kinetic energy matrix for electron
    T = np.zeros( (Nr,Nr) )
    for rj in range( Nr ):
        for rk in range( Nr ):
            if ( rj == rk ):
                T[rj,rk] = np.pi**2 / 3
            else:
                T[rj,rk] = (-1)**(rj-rk) * 2 / (rj-rk)**2
    T /= 2 * dr**2

    R_LIST      = np.arange( -8,8+0.1,0.1 )
    E_LIST      = np.zeros( (len(R_LIST), NEL) )
    U_LIST      = np.zeros( (len(R_LIST), Nr,NEL) )
    DIPOLE      = np.zeros( (len(R_LIST), NEL,NEL) )
    GRAD_E      = np.zeros( (len(R_LIST), NEL) )
    NACR        = np.zeros( (len(R_LIST), NEL,NEL) )
    GRAD_DIPOLE = np.zeros( (len(R_LIST), NEL,NEL) )

    def correct_phase( Unew, Uold=None ):
        PHASE = np.einsum("xj,xj->j", Unew, Uold)
        for n in range( NEL ):
            f = np.cos( np.angle(PHASE[n]) )
            Unew[:,n] = f * Unew[:,n] # phase correction
        return Unew

    dR = 1e-5
    print( "\tCalculating Hel on coarse grid..." )
    for Ri,R in enumerate( R_LIST ):
        e,u                 = np.linalg.eigh( T + get_V(R) )
        E_LIST[Ri,:]        = e[:NEL]
        U_LIST[Ri,:,:]      = u[:,:NEL]
        if ( Ri > 0 ):
            U_LIST[Ri,:,:]  = correct_phase( U_LIST[Ri,:,:], Uold=U_LIST[Ri-1,:,:] )
        DIPOLE[Ri,:,:]      = np.einsum( "ri,r,rj->ij", U_LIST[Ri,:,:], rgrid[:], U_LIST[Ri,:,:] )
        eplus,uplus         = np.linalg.eigh( T + get_V(R+dR) )
        uplus               = correct_phase( uplus[:,:NEL], Uold=U_LIST[Ri,:,:] )
        NACR[Ri,:,:]        = np.einsum( "ri,rj->ij", U_LIST[Ri,:,:], uplus[:,:NEL] ) / dR
        GRAD_E[Ri,:]        = (eplus[:NEL]-e[:NEL]) / dR
        dipoleplus          = np.einsum( "ri,r,rj->ij", uplus[:,:NEL], rgrid[:], uplus[:,:NEL] )
        GRAD_DIPOLE[Ri,:,:] = (dipoleplus[:NEL,:NEL] - DIPOLE[Ri,:NEL,:NEL]) / dR

    # Set NACR diagonals to zero
    for Ri in range( len(R_LIST) ):
        for n in range( NEL ):
            NACR[Ri,n,n] = 0.0

    print( "\tInterpolating H_el..." )
    # Interpolate each quantity over first axis using cubic splines
    from scipy.interpolate import interp1d
    E_interp           = interp1d( R_LIST, E_LIST,      kind='cubic', axis=0 )
    U_LIST_interp      = interp1d( R_LIST, U_LIST,      kind='cubic', axis=0 )
    DIPOLE_interp      = interp1d( R_LIST, DIPOLE,      kind='cubic', axis=0 )
    GRAD_E_interp      = interp1d( R_LIST, GRAD_E,      kind='cubic', axis=0 )
    NACR_interp        = interp1d( R_LIST, NACR,        kind='cubic', axis=0 )
    GRAD_DIPOLE_interp = interp1d( R_LIST, GRAD_DIPOLE, kind='cubic', axis=0 )

    # Plot each quantity in a separate plot as a function of R_LIST and a fine R grid
    R_FINE = np.arange( -6,6+0.01,0.01 )
    plt.plot( R_LIST, E_LIST[:,0] - np.min(E_LIST[:,0]), "o", label='S0 (exact)' )
    plt.plot( R_LIST, E_LIST[:,1] - np.min(E_LIST[:,0]), "o", label='S1 (exact)' )
    plt.plot( R_FINE, E_interp(R_FINE)[:,0] - np.min(E_LIST[:,0]), label='S0 (interpolated)' )
    plt.plot( R_FINE, E_interp(R_FINE)[:,1] - np.min(E_LIST[:,0]), label='S1 (interpolated)' )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/E_Hel.jpg", dpi=300 )
    plt.clf()
    plt.close()

    plt.plot( R_LIST, DIPOLE[:,0,0], "o", label='00 (exact)' )
    plt.plot( R_LIST, DIPOLE[:,0,1], "o", label='01 (exact)' )
    plt.plot( R_LIST, DIPOLE[:,1,0], "o", label='10 (exact)' )
    plt.plot( R_LIST, DIPOLE[:,1,1], "o", label='11 (exact)' )
    plt.plot( R_FINE, DIPOLE_interp(R_FINE)[:,0,0], label='00 (interpolated)' )
    plt.plot( R_FINE, DIPOLE_interp(R_FINE)[:,0,1], label='01 (interpolated)' )
    plt.plot( R_FINE, DIPOLE_interp(R_FINE)[:,1,0], label='10 (interpolated)' )
    plt.plot( R_FINE, DIPOLE_interp(R_FINE)[:,1,1], label='11 (interpolated)' )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/DIPOLE_Hel.jpg", dpi=300 )
    plt.clf()
    plt.close()

    plt.plot( R_LIST, GRAD_E[:,0], "o", label='S0 (exact)' )
    plt.plot( R_LIST, GRAD_E[:,1], "o", label='S1 (exact)' )
    plt.plot( R_FINE, GRAD_E_interp(R_FINE)[:,0], label='S0 (interpolated)' )
    plt.plot( R_FINE, GRAD_E_interp(R_FINE)[:,1], label='S1 (interpolated)' )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/GRAD_E_Hel.jpg", dpi=300 )
    plt.clf()
    plt.close()

    plt.plot( R_LIST, NACR[:,0,1], "o", label='S0-S1 (exact)' )
    plt.plot( R_LIST, NACR[:,1,0], "o", label='S1-S0 (exact)' )
    plt.plot( R_FINE, NACR_interp(R_FINE)[:,0,1], label='S0-S1 (interpolated)' )
    plt.plot( R_FINE, NACR_interp(R_FINE)[:,1,0], label='S1-S0 (interpolated)' )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/NACR_Hel.jpg", dpi=300 )
    plt.clf()
    plt.close()

    plt.plot( R_LIST, GRAD_DIPOLE[:,0,0], "o", label='00 (exact)' )
    plt.plot( R_LIST, GRAD_DIPOLE[:,0,1], "o", label='01 (exact)' )
    plt.plot( R_LIST, GRAD_DIPOLE[:,1,0], "o", label='10 (exact)' )
    plt.plot( R_LIST, GRAD_DIPOLE[:,1,1], "o", label='11 (exact)' )
    plt.plot( R_FINE, GRAD_DIPOLE_interp(R_FINE)[:,0,0], label='00 (interpolated)' )
    plt.plot( R_FINE, GRAD_DIPOLE_interp(R_FINE)[:,0,1], label='01 (interpolated)' )
    plt.plot( R_FINE, GRAD_DIPOLE_interp(R_FINE)[:,1,0], label='10 (interpolated)' )
    plt.plot( R_FINE, GRAD_DIPOLE_interp(R_FINE)[:,1,1], label='11 (interpolated)' )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/GRAD_DIPOLE_Hel.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # Make dictionary to store all data
    MOL_DATA = {}
    MOL_DATA["ENERGY"]      = E_interp
    MOL_DATA["U"]           = U_LIST_interp
    MOL_DATA["DIPOLE"]      = DIPOLE_interp
    MOL_DATA["GRAD_E"]      = GRAD_E_interp
    MOL_DATA["NACR"]        = NACR_interp
    MOL_DATA["GRAD_DIPOLE"] = GRAD_DIPOLE_interp
    return MOL_DATA

def get_H_TC( R, MOL_DATA ):
    E = MOL_DATA["ENERGY"](R) # (mol,el)
    D = MOL_DATA["DIPOLE"](R) # (mol,el,el)

    # # Do CS shift -- Equivalent to neglecting the H_TC[0,-1] and H_TC[-1,0] elements
    # for A in range( NMOL ):
    #     D[A,:,:] = D[A,:,:] - np.eye(NEL) * D[A,0,0]

    H_TC        = np.zeros( (1+NMOL+1,1+NMOL+1) )
    H_TC[0,0]   = np.sum( E[:,0] )
    H_TC[-1,-1] = H_TC[0,0] + WC
    for A in range( NMOL ):
        H_TC[1+A,1+A] = H_TC[0,0] - E[A,0] + E[A,1]
        H_TC[1+A,-1]   = WC * A0 * D[A,0,1] / np.sqrt(NMOL)
        H_TC[-1,1+A]   = WC * A0 * D[A,0,1] / np.sqrt(NMOL)

    E_TC, U_TC = np.linalg.eigh( H_TC )
    return H_TC, E_TC, U_TC

def timer( func ):
    def wrapper( *args, **kwargs ):
        start  = time()
        result = func( *args, **kwargs )
        end    = time()
        print( "Function %s took %1.4f s." % (func.__name__, end-start) )
        return result
    return wrapper

# @timer
def get_TC_Force( R, Z, U_TC, MOL_DATA ):
    RHO         = np.einsum( "j,k->jk", Z.conj(), Z )
    E           = MOL_DATA["ENERGY"](R) # (mol,el)
    GRAD_E      = MOL_DATA["GRAD_E"](R) # (mol,el)
    GRAD_DIPOLE = MOL_DATA["GRAD_DIPOLE"](R) # (mol,el,el)
    NACR        = MOL_DATA["NACR"](R) # (mol,el,el)
    F           = np.zeros( (NMOL, NPOL, NPOL) )

    for P in range( NPOL ):
        F[:,P,P]     = -GRAD_E[:,0]

    for A in range( NMOL ):
        F[A,1+A,1+A] = -GRAD_E[A,1]
        F[A,-1,1+A]  = -WC * A0 * GRAD_DIPOLE[A,0,1] / np.sqrt(NMOL)
        F[A,1+A,-1]  = -WC * A0 * GRAD_DIPOLE[A,1,0] / np.sqrt(NMOL)
        F[A,0,1+A]   = -NACR[A,0,1] * ( E[A,1] - E[A,0] )
        F[A,1+A,0]   = F[A,0,1+A]
    
    F = np.einsum("aJ,Rab,bK->RJK", U_TC, F, U_TC, optimize=True)
    F = np.einsum("RJK,JK->R", F, RHO.real, optimize=True)
    return F

# @timer
@njit()
def get_S_el( U1, U0 ):
    S = np.zeros( (NMOL, U0.shape[-1], U0.shape[-1]), dtype=np.complex128 )
    for A in range( NMOL ):
        S[A,:,:] = U1[A,:,:].T @ U0[A,:,:] # S = np.einsum("Axj,Axk->Ajk", U1, U0)
        u,s,v = np.linalg.svd( S[A,:,:] )
        S[A,:,:] = u @ v
    
    S_el      = np.zeros( (NPOL, NPOL), dtype=np.complex128 )
    S_el[0,0] = np.prod( S[:,0,0] )
    S_el[-1,-1] = 1.0
    for A in range( NMOL ):
        S_el[1+A,1+A] = S_el[0,0] * S[A,1,1] / S[A,0,0]
        S_el[0,1+A]   = S_el[0,0] * S[A,0,1] / S[A,0,0]
        S_el[1+A,0]   = S_el[0,0] * S[A,1,0] / S[A,0,0]

    return S_el

# @timer
@njit()
def correct_phase( Unew, Uold=None ):
    PHASE = np.sum(Unew.T.conj() * Uold, axis=0) # np.einsum("xj,xj->j", Unew, Uold)
    for n in range( NEL ):
        f = np.cos( np.angle(PHASE[n]) )
        Unew[:,n] = f * Unew[:,n] # phase correction
    return Unew

#@timer
@njit()
def do_electronic_propagation( step, E_TC_1, U_TC_1, U_TC_0, U1, U0, Zt_pol, Zt_adF ):
    S_POL          = U_TC_1.T @ U_TC_0 # np.einsum("xj,xk->jk", U_TC_1, U_TC_0)
    S_el           = get_S_el( U1, U0 ) # <t1|t0>
    S_el[:,:]      = U_TC_0.T @ S_el @ U_TC_0 # S_el = np.einsum("xj,xy,yk->jk", U_TC_0, S_el, U_TC_0) # <t1|t0>

    Zt_pol[step,:] = S_POL @ Zt_pol[step-1,:] # np.einsum("jk,k->j", S_POL, Zt_pol[step-1,:])
    Zt_pol[step,:] = S_el  @ Zt_pol[step,:]   # np.einsum("jk,k->j", S_el, Zt_pol[step,:])
    Zt_pol[step,:] = np.exp(-1j * E_TC_1 * dtN) * Zt_pol[step,:]
    Zt_adF[step,:] = U_TC_1 @ Zt_pol[step,:] # np.einsum("FP,P->F", U_TC_1, Zt_pol[step,:]) # POL to adFock

    return Zt_pol, Zt_adF

def do_Ehrenfest( R0, V0, MOL_DATA ):

    Rt     = np.zeros( (NSTEPS,NMOL) )
    Vt     = np.zeros( (NSTEPS,NMOL) )
    Zt_pol = np.zeros( (NSTEPS,NPOL), dtype=np.complex128 )
    Zt_adF = np.zeros( (NSTEPS,NPOL), dtype=np.complex128 )
    Et     = np.zeros( (NSTEPS,3) )
    EPOLt  = np.zeros( (NSTEPS,NPOL) )
    PHOT   = np.zeros( (NSTEPS,NPOL) )

    Rt[0,:] = R0
    Vt[0,:] = V0

    # Do first polaritonic structure calculation
    H_TC_0, E_TC_0, U_TC_0 = get_H_TC( Rt[0,:], MOL_DATA )
    EPOLt[0,:] = E_TC_0

    if ( INIT_BASIS == "POL" ):
        Zt_pol[0,INIT_STATE] = 1.0 # {INIT_BASIS} BASIS -- see get_Globals()
        Zt_adF[0,:] = np.einsum("FP,P->F", U_TC_0, Zt_pol[0,:]) # POL to adFock
    elif ( INIT_BASIS == "adFock" ):
        Zt_adF[0,INIT_STATE] = 1.0 # {INIT_BASIS} BASIS -- see get_Globals()
        Zt_pol[0,:] = np.einsum("FP,F->P", U_TC_0, Zt_adF[0,:]) # adFock to POL

    F0  = get_TC_Force( Rt[0,:], Zt_pol[0,:], U_TC_0, MOL_DATA )
    RF0 = np.sqrt(2 * kT * L_COEFF / dtN) * np.random.normal(0, 1, size=NMOL)

    Et[0,0] = np.sum( E_TC_0 * np.abs(Zt_pol[0,:])**2 ) # Potential Energy
    Et[0,1] = 0.500 * MASS * np.sum(Vt[0,:]**2) # Kinetic Energy
    Et[0,2] = Et[0,0] + Et[0,1]
    N_OP_adF     = np.zeros(NPOL)
    N_OP_adF[-1] = 1.0
    PHOT[0,:]  = np.abs(U_TC_0[-1,:])**2

    for step in range( 1, NSTEPS ):
        #if ( step == 2 or step % 100 == 0 ):
        #    print( "Step %d of %d" % (step, NSTEPS) )
        
        Vt[step,:]             = Vt[step-1] + 0.5 * dtN * (F0 - L_COEFF * Vt[step-1] + RF0 ) / MASS
        Rt[step,:]             = Rt[step-1] +  + dtN * Vt[step]
        H_TC_1, E_TC_1, U_TC_1 = get_H_TC( Rt[step,:], MOL_DATA )
        U_TC_1                 = correct_phase( U_TC_1, Uold=U_TC_0 )
        Zt_pol, Zt_adF         = do_electronic_propagation( step, E_TC_1, U_TC_1.astype(np.complex128), U_TC_0.astype(np.complex128), MOL_DATA["U"](Rt[step,:]).astype(np.complex128), MOL_DATA["U"](Rt[step-1,:]).astype(np.complex128), Zt_pol, Zt_adF )
        F1                     = get_TC_Force( Rt[step], Zt_pol[step], U_TC_1, MOL_DATA )
        RF1                    = np.sqrt(2 * kT * L_COEFF / dtN) * np.random.normal(0, 1, size=NMOL)
        Vt[step,:]             = Vt[step] + 0.5 * dtN * (F1 - L_COEFF * Vt[step] + RF1 ) / MASS

        Et[step,0]     = np.sum( E_TC_1 * np.abs(Zt_pol[step,:])**2 ) # Potential Energy
        Et[step,1]     = 0.500 * MASS * np.sum(Vt[step,:]**2) # Kinetic Energy
        Et[step,2]     = Et[step,0] + Et[step,1]
        EPOLt[step,:]  = E_TC_1
        PHOT[step,:]   = np.abs(U_TC_1[-1,:])**2
        
        F0     = F1
        RF0    = RF1
        U_TC_0 = U_TC_1
        E_TC_0 = E_TC_1
        H_TC_0 = H_TC_1

    return Rt, Vt, Zt_pol, Zt_adF, Et,EPOLt, PHOT

def plot_POL_PES( MOL_DATA, Rt=None, Vt=None ):
    print("\tPlotting PES...")
    R_LIST = np.arange( -6,6+0.01,0.01 )
    E_TC   = np.zeros( (len(R_LIST), NPOL) )
    R_TMP  = np.zeros( NMOL )
    R_TMP[:] = -2.6
    for Ri,R in enumerate( R_LIST ):
        R_TMP[0] = R
        _, E_TC[Ri,:], _ = get_H_TC( R_TMP, MOL_DATA )
    
    for state in range( NPOL ):
        if ( state % 2 == 0 ):
            plt.plot( R_LIST, E_TC[:,state] - np.min(E_TC[:,0]), "-", label=(NPOL<10)*('P%d' % state) )
        else:
            plt.plot( R_LIST, E_TC[:,state] - np.min(E_TC[:,0]), "--", label=(NPOL<10)*('P%d' % state) )

    # Calculate Boltzmann distribution from the first PES
    R_BOLTZMANN = np.zeros( (len(R_LIST)) )
    for Ri,R in enumerate( R_LIST ):
        R_TMP[0] = R
        E_R = E_TC[Ri,0] - np.min(E_TC[:,0])
        R_BOLTZMANN[Ri] = np.exp(-E_R/kT) # kT defined in globals
    
    # Calculate Maxwell-Boltzmann velocity distribution from the temperature and mass
    V_LIST              = np.linspace(0,0.003,5000)
    V_BOLTZMANN         = np.exp( -MASS * V_LIST**2 / 2 / kT ) # V_LIST**2 * np.exp( -MASS * V_LIST**2 / kT )
    R_BOLTZMANN         = R_BOLTZMANN / np.sum(R_BOLTZMANN)
    V_BOLTZMANN         = V_BOLTZMANN / np.sum(V_BOLTZMANN)

    SCALE_FACTOR = WC/2 / np.max(R_BOLTZMANN)
    plt.plot( R_LIST, SCALE_FACTOR * R_BOLTZMANN, "-", c="black", label='$\\mathcal{P} \\sim \\mathrm{exp}[-E_0/kT]$' )
    plt.xlabel( "R", fontsize=15 )
    plt.ylabel( "Energy (a.u.)", fontsize=15 )    
    if ( Rt is not None ):
        bins, edges = np.histogram( Rt.flatten(), bins=100 )
        edges = (edges[:-1] + edges[1:])/2
        SCALE_FACTOR = WC/2 / np.max(bins)
        plt.plot( edges, SCALE_FACTOR * bins, "--", color="green", label="Langevin Distr." )

    # Do nuclear DVR
    NR   = 1001
    RDVR = np.linspace( -6, 6, NR )
    dR   = RDVR[1] - RDVR[0]
    T    = np.zeros( (NR,NR) )
    for Ri in range( NR ):
        for Rj in range( NR ):
            if ( Ri == Rj ):
                T[Ri,Ri] = np.pi**2 / 3
            else:
                T[Ri,Rj] = (-1)**(Ri-Rj) * 2 / (Ri-Rj)**2
    T /= 2 * dR**2 * MASS
    V = np.zeros( (NR) )
    R_TMP    = np.zeros( NMOL )
    R_TMP[:] = -2.6
    for Ri,R in enumerate( RDVR ):
        R_TMP[0]   = R
        _, ENOW, _ = get_H_TC( R_TMP, MOL_DATA )
        V[Ri]      = ENOW[0]
    E, U = np.linalg.eigh( T + np.diag(V) )
    SCALE_FACTOR = WC/2 / np.max(np.abs(U[:,0])**2)
    plt.plot( RDVR, SCALE_FACTOR * np.abs(U[:,0])**2, "-.", color="red", label="DVR" )

    plt.legend()
    plt.tight_layout()
    if ( Rt is not None ):
        plt.savefig( f"{DATA_DIR}/E_TC_with_R_histogram.jpg", dpi=300 )
    else:
        plt.savefig( f"{DATA_DIR}/E_TC.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # Make velcoity histogram if Vt is not None
    if ( Vt is not None ):
        plt.plot( V_LIST, V_BOLTZMANN / np.max(V_BOLTZMANN), "-", label='$\\mathcal{P} \\sim v^2 \\mathrm{exp}[-mv^2/2kT]$' )
        bins, edges = np.histogram( np.abs(Vt).flatten(), bins=100 )
        edges = (edges[:-1] + edges[1:])/2
        plt.plot( edges, bins / np.max(bins), "--", color="green", label="Langevin Distr." )
        plt.xlabel( "V", fontsize=15 )
        plt.ylabel( "Probability", fontsize=15 )
        plt.xlim(0.0,0.005)
        plt.legend()
        plt.savefig( f"{DATA_DIR}/V_HISTOGRAM.jpg", dpi=300 )
        plt.clf()
        plt.close()

    return R_LIST, R_BOLTZMANN, V_LIST, V_BOLTZMANN




if ( __name__ == "__main__" ):
    get_Globals()   
    MOL_DATA                                 = interpolate_Hel()
    R_LIST, R_BOLTZMANN, V_LIST, V_BOLTZMANN = plot_POL_PES( MOL_DATA )

    Rt     = np.zeros( (NTRAJ,NSTEPS,NMOL) )
    Vt     = np.zeros( (NTRAJ,NSTEPS,NMOL) )
    Zt_pol = np.zeros( (NTRAJ,NSTEPS,NPOL), dtype=np.complex128 )
    Zt_adF = np.zeros( (NTRAJ,NSTEPS,NPOL), dtype=np.complex128 )
    Et     = np.zeros( (NTRAJ,NSTEPS,3) )
    EPOLt  = np.zeros( (NTRAJ,NSTEPS,NPOL) )
    PHOT   = np.zeros( (NTRAJ,NSTEPS,NPOL) )

    for traj in range( NTRAJ ):
        print( "Trajectory %d of %d" % (traj, NTRAJ) )
        R0             = np.random.choice(R_LIST, size=NMOL, p=R_BOLTZMANN)
        V0             = np.random.choice(V_LIST, size=NMOL, p=V_BOLTZMANN)
        T  = 2 * (0.500 * MASS * np.sum(V0**2)) / NMOL / (0.025 / 300 / 27.2114)
        print("\tInitial Temperature: %1.2f K" % T )
        Rt[traj], Vt[traj], Zt_pol[traj], Zt_adF[traj], Et[traj], EPOLt[traj], PHOT[traj] = do_Ehrenfest( R0, V0, MOL_DATA )

    # Plot a histogram of the positions
    plot_POL_PES( MOL_DATA, Rt=Rt, Vt=Vt )

    # Save data
    np.save( f"{DATA_DIR}/R.npy", Rt )
    np.save( f"{DATA_DIR}/V.npy", Vt )
    np.save( f"{DATA_DIR}/Z_POL.npy", Zt_pol )
    np.save( f"{DATA_DIR}/Z_adF.npy", Zt_adF )
    np.save( f"{DATA_DIR}/E.npy", Et )
    np.save( f"{DATA_DIR}/EPOL.npy", EPOLt )
    np.save( f"{DATA_DIR}/PHOT.npy", PHOT )

    TEMP  = 2 * Et[:,:,1] / NMOL / (0.025 / 300 / 27.2114)  # in Kelvin
    TEMP  = np.average(TEMP, axis=0)

    # Average over trajectories
    Rt    = np.average( Rt   , axis=0 )
    Vt    = np.average( Vt   , axis=0 )
    Et    = np.average( Et   , axis=0 )
    #EPOLt = np.average( EPOLt, axis=0 )

    # Plot Rt
    for mol in range( NMOL ):
        plt.plot( TIME, Rt[:,mol], "-"*(mol%2==0)+"--"*(mol%2!=0), label=(NMOL<10)*("Molecule %d" % mol) )
    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "Position (a.u.)", fontsize=15 )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/R.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # Plot Vt
    for mol in range( NMOL ):
        plt.plot( TIME, Vt[:,mol], "-"*(mol%2==0)+"--"*(mol%2!=0), label=(NMOL<10)*("Molecule %d" % mol) )
    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "Velocity (a.u.)", fontsize=15 )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/V.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # Plot Polaritonic Population
    POP = np.abs(Zt_pol)**2
    POP = np.average(POP, axis=0)
    plt.plot( TIME, np.sum(POP[:,:],axis=-1), "-", alpha=0.5, c='black', lw=6 )
    plt.plot( TIME, POP[:,0], lw=4, alpha=0.5, label="P0" )
    plt.plot( TIME, POP[:,1], lw=4, alpha=0.5, label="LP" )
    if ( NMOL >= 2 ):
        plt.plot( TIME, np.sum( POP[:,2:-1], axis=-1 ), lw=4, alpha=0.5, label="'Dark'" )
    plt.plot( TIME, POP[:,-1], lw=4, alpha=0.5, label="UP" )
    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "Population (a.u.)", fontsize=15 )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/POP_POL.jpg", dpi=300 )
    # Save another version with logarithmic y-axis
    plt.yscale('log')
    plt.ylim(1e-4,1)
    plt.savefig( f"{DATA_DIR}/POP_POL_log.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # Plot adiabatic-Fock population
    POP = np.abs(Zt_adF)**2
    POP = np.average(POP, axis=0)
    plt.plot( TIME, np.sum(POP[:,:],axis=-1), "-", alpha=0.5, c='black', lw=6 )
    plt.plot( TIME, POP[:,0], "-", lw=4, alpha=0.5, label="G0" )
    plt.plot( TIME, np.sum(POP[:,1:-1],axis=-1), "-", lw=4, alpha=0.5, label="Exciton" )
    plt.plot( TIME, POP[:,-1], "-", lw=4, alpha=0.5, label="G1" )

    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "Population (a.u.)", fontsize=15 )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/POP_adF.jpg", dpi=300 )
    # Save another version with logarithmic y-axis
    plt.yscale('log')
    plt.ylim(1e-4,1)
    plt.savefig( f"{DATA_DIR}/POP_adF_log.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # Plot Et
    plt.plot( TIME, Et[:,0], "-", label="EPOT" )
    plt.plot( TIME, Et[:,1], "-", label="EKIN" )
    plt.plot( TIME, Et[:,2], "--", label="ETOT" )
    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "Population (a.u.)", fontsize=15 )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/EPOT_EKIN_ETOT.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # Plot total energy
    plt.plot( TIME, Et[:,2], "--", label="ETOT" )
    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "Population (a.u.)", fontsize=15 )
    plt.legend()
    plt.savefig( f"{DATA_DIR}/ETOT.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # Plot the average photonic character
    PHOT_wfn = np.abs(Zt_adF[:,:,-1])**2
    PHOT_wfn = np.average(PHOT_wfn, axis=0)
    plt.plot( TIME, PHOT_wfn, "-" )
    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "Photonic Character, $\\langle \\hat{a}^\\dagger \\hat{a} \\rangle$", fontsize=15 )
    plt.savefig( f"{DATA_DIR}/PHOTONIC.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # Plot the temperature
    plt.plot( TIME, TEMP, "-" )
    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "Temperature (K)", fontsize=15 )
    plt.savefig( f"{DATA_DIR}/TEMPERATURE.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # # Plot the polariton energies along R0 coordinate
    # time_color = np.linspace( 0, 1, NSTEPS )
    # cmap = plt.get_cmap('terrain')
    # for p in range( NPOL ):
    #     plt.scatter( Rt[:,0], EPOLt[:,p] - EPOLt[0,0], s=1, color=cmap(time_color[:]) )
    # plt.xlabel( "Position, $R_0$ (a.u.)", fontsize=15 )
    # plt.ylabel( "Energy (a.u.)", fontsize=15 )
    # plt.legend()
    # plt.savefig( f"{DATA_DIR}/PES_R0.jpg", dpi=300 )
    # plt.clf()
    # plt.close()

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                        left=0.12, right=0.95, bottom=0.12, top=0.95,
                        wspace=0.05)
    ax = fig.add_subplot( gs[0, 0] )
    ax_hist = fig.add_subplot( gs[0, 1] )
    EPOLt *= 27.2114
    for p in range( 1, NPOL ): # Plot first trajectory only
        ax.plot( TIME/41.341, EPOLt[0,:,p] - EPOLt[0,:,0], c="black", alpha=0.25, lw=1 )
    ax.plot( TIME/41.341, np.average(EPOLt[:,:,-1] - EPOLt[:,:,0], axis=0), c="blue", lw=4, alpha=0.75, label="$\\langle$UP$\\rangle$" ) # AVE UP
    ax.plot( TIME/41.341, np.average(EPOLt[:,:,2:-1] - EPOLt[:,:,0][:,:,None], axis=(0,2)), lw=4, c="green", alpha=0.75, label="$\\langle$MP$\\rangle$" ) # AVE UP
    ax.plot( TIME/41.341, np.average(EPOLt[:,:,1]  - EPOLt[:,:,0], axis=0), c="red", lw=4, alpha=0.75, label="$\\langle$LP$\\rangle$" ) # AVE UP
    ax.plot( TIME/41.341, TIME*0 + WC*27.2114, "--", alpha=0.75, lw=2, c="orange", label="$\\omega_\\mathrm{c}$" ) # WC

    ax.set_ylabel( "Transition Energy (eV)", fontsize=15 )
    ax.set_xlabel( "Time (fs)", fontsize=15 )

    NBINS = 100
    EMIN  = np.min(EPOLt[:,:,1:] - EPOLt[:,:,0][:,:,None])
    EMAX  = np.max(EPOLt[:,:,1:] - EPOLt[:,:,0][:,:,None])
    EBIN  = np.linspace( EMIN, EMAX, NBINS )
    DOS  = np.zeros( NBINS-1 )
    TM   = np.zeros( NBINS-1 )
    for b in range( NBINS-1 ):
        for traj in range( NTRAJ ):
            for ti in range( NSTEPS ):
                for p in range( 1, NPOL ):
                    if ( EPOLt[traj,ti,p] - EPOLt[traj,ti,0] >= EBIN[b] and EPOLt[traj,ti,p] - EPOLt[traj,ti,0] < EBIN[b+1] ):
                        DOS[b] += 1.0
                        TM[b]  += PHOT[traj,ti,p]
    EBIN = (EBIN[:-1] + EBIN[1:])/2
    ax_hist.semilogx( DOS/np.max(DOS), EBIN, c="black" )
    ax_hist.semilogx( TM/np.max(TM), EBIN, c="red" )
    ax.set_xlim( 0, TIME[-2]/41.341 )
    ax_hist.set_xlim( 1e-2, 1 )
    ax_hist.set_ylim( ax.get_ylim() )
    #ax_hist.tick_params(axis="x", labelbottom=False)
    ax_hist.tick_params(axis="y", labelleft=False)
    #ax_hist.set_xlabel( "log[SPEC]", fontsize=15 )
    #ax_hist.tick_params(axis='y', labelcolor='k')
    #ax_hist.set_ylabel( "DOS (arb.)", fontsize=15 )
    ax.legend()
    plt.savefig( f"{DATA_DIR}/PES_time.jpg", dpi=300 )
    plt.clf()
    plt.close()



    # Calculate the inverse participation ratio
    IPR = np.zeros( (NTRAJ,NSTEPS) )
    for traj in range( NTRAJ ):
        for step in range( NSTEPS ):
            WFN  = Zt_adF[traj,step,1:-1] # Exclude G0 and G1. Keep only local mol excitations.
            PROB = np.abs(WFN)*2
            NORM = np.sum( PROB )
            PROB = PROB / NORM
            IPR[traj,step] = 1 / np.sum( PROB**2 )
    IPR = np.average( IPR, axis=0 )
    plt.plot( TIME, IPR / NMOL, "-" )
    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "% IPR ($\\frac{IPR}{N_\\mathrm{mol}}$)", fontsize=15 )
    plt.tight_layout()
    plt.savefig( f"{DATA_DIR}/IPR.jpg", dpi=300 )
    plt.clf()
    plt.close()