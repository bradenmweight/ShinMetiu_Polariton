import numpy as np
import matplotlib.pyplot as plt

def get_Globals():
    global NMOL, NEL, NPOL
    NMOL = 10
    NEL  = 2
    NPOL = 1 + NMOL*(NEL-1) + 1

    global TIME, NSTEPS, dtN, MASS
    NSTEPS = 2500
    dtN    = 1
    TIME   = np.arange( 0, NSTEPS*dtN, dtN )
    MASS   = 1836.0

    global A0, wc
    A0 = 0.02
    wc = 0.08

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

    R_LIST      = np.arange( -6,6+0.1,0.1 )
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
    plt.savefig( "E.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # plt.plot( R_LIST, DIPOLE[:,0,0], "o", label='00 (exact)' )
    # plt.plot( R_LIST, DIPOLE[:,0,1], "o", label='01 (exact)' )
    # plt.plot( R_LIST, DIPOLE[:,1,0], "o", label='10 (exact)' )
    # plt.plot( R_LIST, DIPOLE[:,1,1], "o", label='11 (exact)' )
    # plt.plot( R_FINE, DIPOLE_interp(R_FINE)[:,0,0], label='00 (interpolated)' )
    # plt.plot( R_FINE, DIPOLE_interp(R_FINE)[:,0,1], label='01 (interpolated)' )
    # plt.plot( R_FINE, DIPOLE_interp(R_FINE)[:,1,0], label='10 (interpolated)' )
    # plt.plot( R_FINE, DIPOLE_interp(R_FINE)[:,1,1], label='11 (interpolated)' )
    # plt.legend()
    # plt.savefig( "DIPOLE.jpg", dpi=300 )
    # plt.clf()
    # plt.close()

    # plt.plot( R_LIST, GRAD_E[:,0], "o", label='S0 (exact)' )
    # plt.plot( R_LIST, GRAD_E[:,1], "o", label='S1 (exact)' )
    # plt.plot( R_FINE, GRAD_E_interp(R_FINE)[:,0], label='S0 (interpolated)' )
    # plt.plot( R_FINE, GRAD_E_interp(R_FINE)[:,1], label='S1 (interpolated)' )
    # plt.legend()
    # plt.savefig( "GRAD_E.jpg", dpi=300 )
    # plt.clf()
    # plt.close()

    # plt.plot( R_LIST, NACR[:,0,1], "o", label='S0-S1 (exact)' )
    # plt.plot( R_LIST, NACR[:,1,0], "o", label='S1-S0 (exact)' )
    # plt.plot( R_FINE, NACR_interp(R_FINE)[:,0,1], label='S0-S1 (interpolated)' )
    # plt.plot( R_FINE, NACR_interp(R_FINE)[:,1,0], label='S1-S0 (interpolated)' )
    # plt.legend()
    # plt.savefig( "NACR.jpg", dpi=300 )
    # plt.clf()
    # plt.close()

    # plt.plot( R_LIST, GRAD_DIPOLE[:,0,0], "o", label='00 (exact)' )
    # plt.plot( R_LIST, GRAD_DIPOLE[:,0,1], "o", label='01 (exact)' )
    # plt.plot( R_LIST, GRAD_DIPOLE[:,1,0], "o", label='10 (exact)' )
    # plt.plot( R_LIST, GRAD_DIPOLE[:,1,1], "o", label='11 (exact)' )
    # plt.plot( R_FINE, GRAD_DIPOLE_interp(R_FINE)[:,0,0], label='00 (interpolated)' )
    # plt.plot( R_FINE, GRAD_DIPOLE_interp(R_FINE)[:,0,1], label='01 (interpolated)' )
    # plt.plot( R_FINE, GRAD_DIPOLE_interp(R_FINE)[:,1,0], label='10 (interpolated)' )
    # plt.plot( R_FINE, GRAD_DIPOLE_interp(R_FINE)[:,1,1], label='11 (interpolated)' )
    # plt.legend()
    # plt.savefig( "GRAD_DIPOLE.jpg", dpi=300 )
    # plt.clf()
    # plt.close()

    # Make dictionary to store all data
    MOL_DATA = {}
    MOL_DATA["ENERGY"]      = E_interp
    MOL_DATA["U"]           = U_LIST_interp
    MOL_DATA["DIPOLE"]      = DIPOLE_interp
    MOL_DATA["GRAD_E"]      = GRAD_E_interp
    MOL_DATA["NACR"]        = NACR_interp
    MOL_DATA["GRAD_DIPOLE"] = GRAD_DIPOLE_interp
    return MOL_DATA

def H_TC( R, MOL_DATA ):
    E = MOL_DATA["ENERGY"](R) # (mol,el)
    D = MOL_DATA["DIPOLE"](R) # (mol,el,el)

    # # Do CS shift -- Equivalent to neglecting the H_TC[0,-1] and H_TC[-1,0] elements
    # for A in range( NMOL ):
    #     D[A,:,:] = D[A,:,:] - np.eye(NEL) * D[A,0,0]

    H_TC        = np.zeros( (1+NMOL+1,1+NMOL+1) )
    H_TC[0,0]   = np.sum( E[:,0] )
    H_TC[-1,-1] = H_TC[0,0] + wc
    for A in range( NMOL ):
        H_TC[1+A,1+A] = H_TC[0,0] - E[A,0] + E[A,1]
        H_TC[1+A,-1]   = wc * A0 * D[A,0,1]
        H_TC[-1,1+A]   = wc * A0 * D[A,0,1]

    E_TC, U_TC = np.linalg.eigh( H_TC )
    return H_TC, E_TC, U_TC

def get_TC_Force( R, Z, U_TC, MOL_DATA ):
    RHO         = np.einsum( "j,k->jk", Z.conj(), Z )
    E           = MOL_DATA["ENERGY"](R) # (mol,el)
    GRAD_E      = MOL_DATA["GRAD_E"](R) # (mol,el)
    GRAD_DIPOLE = MOL_DATA["GRAD_DIPOLE"](R) # (mol,el,el)
    NACR        = MOL_DATA["NACR"](R) # (mol,el,el)
    F           = np.zeros( (NMOL, NPOL, NPOL) )

    for A in range( NPOL ):
        F[:,A,A]     = -GRAD_E[:,0]

    for A in range( NMOL ):
        F[A,1+A,1+A] = -GRAD_E[A,1]
        F[A,-1,1+A]  = -wc * A0 * GRAD_DIPOLE[A,0,1]
        F[A,1+A,-1]  = -wc * A0 * GRAD_DIPOLE[A,1,0]
        F[A,0,1+A]   = -NACR[A,0,1] * ( E[A,1] - E[A,0] )
        F[A,1+A,0]   = F[A,0,1+A]
    
    F = np.einsum("xj,Rxy,yk->Rjk", U_TC, F, U_TC, optimize=True)
    F = np.einsum("Rjk,jk->R", F.real, RHO, optimize=True)
    return F

def get_S_el( R1, R0, MOL_DATA ):
    U1 = MOL_DATA["U"](R1) # (mol,x,el)
    U0 = MOL_DATA["U"](R0) # (mol,x,el)
    S  = np.einsum("Axj,Axk->Ajk", U1, U0)
    for A in range( NMOL ):
        u,s,v = np.linalg.svd( S[A,:,:] )
        S[A,:,:] = u @ v
    
    S_el      = np.zeros( (NPOL, NPOL) )
    S_el[0,0] = np.prod( S[:,0,0] )
    S_el[-1,-1] = 1.0
    for A in range( NMOL ):
        S_el[1+A,1+A] = S_el[0,0] * S[A,1,1] / S[A,0,0]
        S_el[0,1+A]   = S_el[0,0] * S[A,0,1] / S[A,0,0]
        S_el[1+A,0]   = S_el[0,0] * S[A,1,0] / S[A,0,0]

    return S_el

def do_Ehrenfest( R0, V0, Z0_POL, MOL_DATA ):

    def correct_phase( Unew, Uold=None ):
        PHASE = np.einsum("xj,xj->j", Unew, Uold)
        for n in range( NEL ):
            f = np.cos( np.angle(PHASE[n]) )
            Unew[:,n] = f * Unew[:,n] # phase correction
        return Unew

    Rt = np.zeros( (NSTEPS,NMOL) )
    Vt = np.zeros( (NSTEPS,NMOL) )
    Zt_pol = np.zeros( (NSTEPS,NPOL), dtype=np.complex128 )
    Zt_adF = np.zeros( (NSTEPS,NPOL), dtype=np.complex128 )
    Et = np.zeros( (NSTEPS,3) )

    Rt[0,:] = R0
    Vt[0,:] = V0
    Zt_pol[0,:] = Z0_POL

    # Do first polaritonic structure calculation
    H_TC_0, E_TC_0, U_TC_0 = H_TC( Rt[0,:], MOL_DATA )
    Zt_adF[0,:] = U_TC_0.T @ Zt_pol[0,:]
    F0 = get_TC_Force( Rt[0,:], Zt_pol[0,:], U_TC_0, MOL_DATA )

    Et[0,0] = np.sum( E_TC_0 * np.abs(Zt_pol[0,:])**2 ) # Potential Energy
    Et[0,1] = 0.500 * MASS * np.sum(Vt[0,:]**2) # Kinetic Energy
    Et[0,2] = Et[0,0] + Et[0,1]

    for step in range( 1, NSTEPS ):
        print( "Step %d of %d" % (step, NSTEPS) )

        Rt[step,:]             = Rt[step-1,:] + dtN * Vt[step-1,:] + 0.5 * dtN**2 * F0 / MASS
        H_TC_1, E_TC_1, U_TC_1 = H_TC( Rt[step,:], MOL_DATA )
        U_TC_1                 = correct_phase( U_TC_1, Uold=U_TC_0 )

        S_POL          = np.einsum("xj,xk->jk", U_TC_1, U_TC_0)
        S_el           = get_S_el( Rt[step,:], Rt[step-1,:], MOL_DATA )
        S_el           = np.einsum("xj,xy,yk->jk", U_TC_0, S_el, U_TC_0)

        Zt_pol[step,:] = np.einsum("jk,k->j", S_POL, Zt_pol[step-1,:])
        Zt_pol[step,:] = np.einsum("jk,k->j", S_el, Zt_pol[step,:])
        Zt_pol[step,:] = np.exp(-1j * E_TC_1 * dtN) * Zt_pol[step,:]
        Zt_adF[step,:] = np.einsum("xj,j->x", U_TC_1, Zt_pol[step,:])

        F1         = get_TC_Force( Rt[step,:], Zt_pol[step,:], U_TC_1, MOL_DATA )
        Vt[step,:] = Vt[step-1,:] + 0.5 * dtN * (F0 + F1) / MASS

        Et[step,0] = np.sum( E_TC_1 * np.abs(Zt_pol[step,:])**2 ) # Potential Energy
        Et[step,1] = 0.500 * MASS * np.sum(Vt[step,:]**2) # Kinetic Energy
        Et[step,2] = Et[step,0] + Et[step,1]

        F0     = F1
        U_TC_0 = U_TC_1
        E_TC_0 = E_TC_1
        H_TC_0 = H_TC_1

    return Rt, Vt, Zt_pol, Zt_adF, Et

def plot_POL_PES( MOL_DATA ):
    R_LIST = np.arange( -6,6+0.01,0.01 )
    E_TC   = np.zeros( (len(R_LIST), NPOL) )
    R_TMP  = np.zeros( NMOL )
    R_TMP[:] = -2.5
    for Ri,R in enumerate( R_LIST ):
        R_TMP[0] = R
        _, E_TC[Ri,:], _ = H_TC( R_TMP, MOL_DATA )
    
    for state in range( NPOL ):
        plt.plot( R_LIST, E_TC[:,state], "-", label='P%d' % state )
    plt.xlabel( "R", fontsize=15 )
    plt.ylabel( "Energy (a.u.)", fontsize=15 )
    plt.legend()
    plt.savefig( "E_TC.jpg", dpi=300 )
    plt.clf()
    plt.close()

if ( __name__ == "__main__" ):
    get_Globals()   
    MOL_DATA = interpolate_Hel()
    plot_POL_PES( MOL_DATA )

    R0 = np.array([-4]*NMOL)
    V0 = np.zeros(NMOL)
    Z0 = np.zeros(NPOL); Z0[1] = 1.0 # POL BASIS
    Rt, Vt, Zt_pol, Zt_adF, Et = do_Ehrenfest( R0, V0, Z0, MOL_DATA)

    # Plot Rt
    for mol in range( NMOL ):
        plt.plot( TIME, Rt[:,mol], "-", label="Molecule %d" % mol )
    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "Position (a.u.)", fontsize=15 )
    plt.legend()
    plt.savefig( "R.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # Plot Rt
    POP = np.abs(Zt_pol)**2
    plt.plot( TIME, np.sum(POP[:,:],axis=-1), "-", alpha=0.5, c='black', lw=6, label="TOTAL" )
    for state in range( NPOL ):
        plt.plot( TIME, POP[:,state], "-", label="State %d" % state )
    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "Population (a.u.)", fontsize=15 )
    plt.legend()
    plt.savefig( "POP_POL.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # Plot POP_adF
    POP = np.abs(Zt_adF)**2
    plt.plot( TIME, np.sum(POP[:,:],axis=-1), "-", alpha=0.5, c='black', lw=6, label="TOTAL" )
    for state in range( NPOL ):
        plt.plot( TIME, POP[:,state], "-", label="State %d" % state )
    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "Population (a.u.)", fontsize=15 )
    plt.legend()
    plt.savefig( "POP_adF.jpg", dpi=300 )
    plt.clf()
    plt.close()

    # Plot Et
    plt.plot( TIME, Et[:,0], "-", label="EPOT" )
    plt.plot( TIME, Et[:,1], "-", label="EKIN" )
    plt.plot( TIME, Et[:,2], "--", label="ETOT" )
    plt.xlabel( "Time (a.u.)", fontsize=15 )
    plt.ylabel( "Population (a.u.)", fontsize=15 )
    plt.legend()
    plt.savefig( "Et.jpg", dpi=300 )
    plt.clf()
    plt.close()