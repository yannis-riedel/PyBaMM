
import pybamm
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import pandas as pd
path, _ = os.path.split(os.path.abspath(__file__))
#path = 'C:\\Users\\Vachenauer\\Documents\\SyncToNas\\07_BattLifeBoost_eigen\\04_AP_31\\01_Modelle\\02_PyBaMM\\p2D_Sony_LFP\\PyBaMM\\pybamm\\input\\parameters\\lithium_ion\\data'
# Alt von Vroni _____________________________________________________________________________________________
#def lfp_ocp_delithiation(sto):
#        lfp_curve, (x, y) = pybamm.parameters.process_1D_data("LFP_Charge_for_PyBaMM.csv", path=path)
#        return pybamm.Interpolant(x, y, sto, name=lfp_curve, interpolator="linear")
#def lfp_ocp_lithiation(sto):
#        lfp_curve, (x, y) = pybamm.parameters.process_1D_data("LFP_Discharge_for_PyBaMM.csv", path=path)
#        return pybamm.Interpolant(x, y, sto, name=lfp_curve, interpolator="linear")
#def lfp_ocp_mean(sto):
#        lfp_curve, (x, y) = pybamm.parameters.process_1D_data("LFP_Mean_for_PyBaMM.csv", path=path)
#        return pybamm.Interpolant(x, y, sto, name=lfp_curve, interpolator="linear")
#
def lfp_ocp_delithiation(sto):
    csv_path = os.path.join(path, "data/Sony_LFP_OCP_Delith.csv")
    # CSV einlesen (z. B. mit Kommentarzeilen "#")
    df = pd.read_csv(csv_path, comment="#", names=["sto", "ocp"])

    # Nach Sto sortieren und doppelte Werte mitteln
    df = df.groupby("sto", as_index=False).mean().sort_values("sto")

    # Arrays extrahieren
    x = df["sto"].values
    y = df["ocp"].values

    # Interpolant erzeugen
    ocp_interp = pybamm.Interpolant(
        x, y, [sto], name="LFP_OCP_delithiation", interpolator="cubic", extrapolate=False
    )

    return ocp_interp

def lfp_ocp_lithiation(sto):
    csv_path = os.path.join(path, "data/Sony_LFP_OCP_Lith.csv")
    # CSV einlesen (z. B. mit Kommentarzeilen "#")
    df = pd.read_csv(csv_path, comment="#", names=["sto", "ocp"])

    # Nach Sto sortieren und doppelte Werte mitteln
    df = df.groupby("sto", as_index=False).mean().sort_values("sto")

    # Arrays extrahieren
    x = df["sto"].values
    y = df["ocp"].values

    # Interpolant erzeugen
    ocp_interp = pybamm.Interpolant(
        x, y, [sto], name="LFP_OCP_lithiation", interpolator="cubic", extrapolate=False
    )

    return ocp_interp

def lfp_ocp_mean(sto):
    lfp_curve, (x, y) = pybamm.parameters.process_1D_data("OCP_LFP_mittelwert.csv", path=path)
    return pybamm.Interpolant(x, y, sto, name=lfp_curve, interpolator="cubic", extrapolate=False) 


#
path, _ = os.path.split(os.path.abspath(__file__))
def graphite_ocp_mean(sto):
    name, (x, y) = pybamm.parameters.process_1D_data("OCP_Graphit_mittelwert.csv", path=path)
    return pybamm.Interpolant(x, y, sto, name=name, interpolator="cubic", extrapolate=False) #* pybamm.Scalar(1)

def graphite_ocp_delithiation(sto):
    csv_path = os.path.join(path, "data/Sony_Graphite_OCP_Delith.csv")
    # CSV einlesen (z. B. mit Kommentarzeilen "#")
    df = pd.read_csv(csv_path, comment="#", names=["sto", "ocp"])

    # Nach Sto sortieren und doppelte Werte mitteln
    df = df.groupby("sto", as_index=False).mean().sort_values("sto")

    # Arrays extrahieren
    x = df["sto"].values
    y = df["ocp"].values

    # Interpolant erzeugen
    ocp_interp = pybamm.Interpolant(
        x, y, [sto], name="LFP_OCP_delithiation", interpolator="cubic", extrapolate=False
    )

    return ocp_interp

def graphite_ocp_lithiation(sto):
    csv_path = os.path.join(path, "data/Sony_Graphite_OCP_Lith.csv")
    # CSV einlesen (z. B. mit Kommentarzeilen "#")
    df = pd.read_csv(csv_path, comment="#", names=["sto", "ocp"])

    # Nach Sto sortieren und doppelte Werte mitteln
    df = df.groupby("sto", as_index=False).mean().sort_values("sto")

    # Arrays extrahieren
    x = df["sto"].values
    y = df["ocp"].values

    # Interpolant erzeugen
    ocp_interp = pybamm.Interpolant(
        x, y, [sto], name="LFP_OCP_lithiation", interpolator="cubic", extrapolate=False
    )

    return ocp_interp
#_______________________________________________________________________________________________________________


#%%
def graphite_entropic_change(sto): 
    return ((0.00527 + 3.29927*sto - 91.79326*(sto**2) + 1004.91101*(sto**3) - 5812.27813*(sto**4) + 19329.75490*(sto**5) 
      - 37147.89470*(sto**6) + 38379.18127*(sto**7) - 16515.05308*(sto**8)) * (
        1 - 48.09287*sto + 1017.23480*(sto**2) - 10481.80419*(sto**3) + 59431.30001*(sto**4) - 195881.64880*(sto**5) 
        + 374577.31520*(sto**6) - 385821.16070*(sto**7) + 165705.85970*(sto**8)  
      )**(-1)*1e-3#Gleichung ist fuer mV/K, PyBaMM braucht V/K
      )

def LFP_entropic_change(sto): 
     return(
          1000*(
               -0.35376*(sto**8) + 1.3902*(sto**7) - 2.2585*(sto**6) + 1.9635*(sto**5) - 0.98716*(sto**4) 
               + 0.28857*(sto**3) - 0.046272*(sto**2) + 0.0032158*(sto) - 1.9186e-5
          )*1e-3 #Gleichung ist fuer mV/K, PyBaMM braucht V/K
     )
    
def LFP_exchange_current_density(c_e, c_s_surf, c_s_max,T):
    """
    Exchange-current density for Butler-Volmer reactions between LFP and electrolyte

    References
    ----------
    .. [1] Rumpf Katharina . Dissertation.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    c_s_ref_pos = 22806/2 #[mol/m^3]
    c_l_ref = 1 #[mol/m^3]
    i_0_ref = 1.5403e-3*(1-0*0.7) # potentiell Parameter zum Optimieren 
    
    
    return (
      T/T * i_0_ref * ((c_e/c_l_ref)**0.5) *((c_s_surf/c_s_ref_pos)**0.5) * (((c_s_max-c_s_surf)/(c_s_max-c_s_ref_pos))**0.5)
    )


def graphite_exchange_current_density(c_e, c_s_surf, c_s_max,T):
     A = 915.6903 # Pre Factor [A.m-2]
     Ea = 2.098e4 # Activation Energy [J.mol-1]
     R = 8.314 

     j_0_neg = (c_e/c_e*c_s_surf/c_s_surf*c_s_max/c_s_max)*A*np.exp(-Ea/(R*T))
     return j_0_neg
    
def LFP_diffusivity(sto,T): 
     A = 6.4444e-6 # Pre-Factor [m.s-1]
     Ea = 7.3017e4 # Activation Energy [J.mol-1]
     R = 8.314 # Universal Gas konstant 

     D_s_pos = sto/sto*A*np.exp(-Ea/(R*T))
     return D_s_pos


def SEI_EC_diffusivity(sto): #50% SoC
    D_EC = sto/sto*4.9e-22
    return D_EC

def electrolyte_diffusivity_Valeon2005(c_e, T):
    """
    Diffusivity of LiPF6  as a function of ion concentration. The data
    comes from [1]

    References
    ----------
    .. [1] L.O. Valeon and J.N. Reimers. "Transport Properties of LiPF6-Based Li-Ion Battery Electrolytes,"
    In: Journal of the Electrochemical Society 152.5 (2005), A882-A891.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    c_e_liter = c_e/1000 # umrechnung von mol*m-3 auf mol l-1

    D_c_e = 10**(-4.43 - 54 / (T - 229 - 5 * c_e_liter) - 0.22 * c_e_liter) * 1e-4

    return D_c_e


def thermodynamic_factor_valoen(c_e, T):
    """
    1 + dlnf/dlnc -> now renamed to thermodynamic factor

    References
    ----------
    .. Dissertation Katharina rumpf

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    c_e_liter = c_e /1000 # umrechnung von mol*m-3 auf mol l-1
    t_plus = 0.38

    thermo_factor = 0.6 * (0.601-0.24*(c_e_liter)**0.5 + 0.982*(1-0.0052*(T-298.15))*(c_e_liter)**1.5)/(1-t_plus)#- 1*0.6

    return thermo_factor


def electrolyte_conductivity_Valeon2005(c_e, T):
    """
    Conductivity of LiPF6  as a function of ion concentration. The data
    comes from [1]

    References
    ----------
    .. [1] L.O. Valeon and J.N. Reimers. "Transport Properties of LiPF6-Based Li-Ion Battery Electrolytes,"
    In: Journal of the Electrochemical Society 152.5 (2005), A882-A891.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    c_e_liter = c_e /1000 # umrechnung von mol*m-3 auf mol l-1
    #kappa_e = (
    #    0.8 * c_e_liter/10 * (-10.5 + 0.074*T - 6.96*1e-5*(T**2) + 0.668*c_e_liter - 0.0178*c_e_liter*T 
    #              + 2.8*1e-5*c_e_liter*(T**2) + 0.494*(c_e_liter**2) - 8.86*1e-4*(c_e_liter**2)*T)**2
    #)
   
    kappa_e = c_e_liter *((-10.5 + 0.074*T - 6.96e-5*(T**2) + 0.668*c_e_liter - 0.0178*c_e_liter*T + 2.8e-5*c_e_liter*(T**2) + 0.494*(c_e_liter**2)-8.86e-4*(c_e_liter**2)*T)**2)*0.1
    # Nyman et al. (2008) does not provide temperature dependence

    return kappa_e




# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for an A123 LFP cell, from the paper

        Michael J. Lain, James Brandon, and Emma Kendrick. Design strategies for high
        power vs. high energy lithium ion cells. Batteries, 5(4):64, 2019.
        doi:10.3390/batteries5040064.

    LG M50 Graphite negative electrode parameters
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Parameters for negative electrode (graphite) and separator are from the paper

        Chang-Hui Chen, Ferran Brosa Planella, Kieran O'Regan, Dominika Gastol, W.
        Dhammika Widanage, and Emma Kendrick. Development of Experimental Techniques for
        Parameterization of Multi-scale Lithium-ion Battery Models. Journal of The
        Electrochemical Society, 167(8):080534, 2020. doi:10.1149/1945-7111/ab9050.

    and references therein.

    Parameters for positive electrode (LFP) are from the paper

        Eric Prada, D. Di Domenico, Y. Creff, J. Bernard, Valérie Sauvant-Moynot, and
        François Huet. A simplified electrochemical and thermal aging model of
        LiFePO4-graphite Li-ion batteries: power and capacity fade simulations. Journal
        of The Electrochemical Society, 160(4):A616, 2013. doi:10.1149/2.053304jes.


    and references therein. The functions used for OCP and exchange-current density are
    from separate references (documented within the functions), to provide better fit to
    data

    Parameters for a LiPF6 electrolyte are from the paper

        Andreas Nyman, Mårten Behm, and Göran Lindbergh. Electrochemical
        characterisation and modelling of the mass transport phenomena in lipf6-ec-emc
        electrolyte. Electrochimica Acta, 53(22):6356-6365, 2008.


    and references therein.
    """

    return {
        "chemistry": "lithium_ion",

        # Cell Parameters
        
        "Electrode height [m]": 0.056,
        "Electrode width [m]": 2.8,
        'Positive electrode height [m]': 0.056,
        'Positive electrode width [m]':2.8, 
        "Negative electrode height [m]": 0.056,
        "Negative electrode width [m]": 2.8,
        "Negative electrode thickness [m]": 58e-06,
        "Separator thickness [m]": 20e-06,
        "Positive electrode thickness [m]": 76e-06,
        "Nominal cell capacity [A.h]": 3,
        "Current function [A]": 3,


        # Current Collector and Tab Parameters (not needed)

        #"Positive current collector thickness [m]": 0,
        #"Negative current collector thickness [m]": 0,
        #"Negative tab width [m]": 0,
        #"Negative tab centre y-coordinate [m]": 0,
        #"Negative tab centre z-coordinate [m]": 0,
        #"Positive tab width [m]": 0.04,
        #"Positive tab centre y-coordinate [m]": 0.147,
        #"Positive tab centre z-coordinate [m]": 0.137,
        #"Negative current collector conductivity [S.m-1]": 58411000.0,
        #"Positive current collector conductivity [S.m-1]": 36914000.0,
        #"Negative current collector density [kg.m-3]": 8960.0,
        #"Positive current collector density [kg.m-3]": 2700.0,
        #"Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        #"Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        #"Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        #"Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        #"Contact resistance [Ohm]": 0,


        # Negative Electrode 
        "Negative electrode conductivity [S.m-1]": 100,
        "Maximum concentration in negative electrode [mol.m-3]": 31370,
        "Negative electrode diffusivity [m2.s-1]": 1e-12,
        #Alt von Vroni: ____________________________________________________________________________
        #"Negative electrode OCP [V]": graphite_ocp_mean,
        #___________________________________________________________________________________________

        # Neu Antonia: _____________________________________________________________________________
        "Negative electrode OCP [V]": graphite_ocp_mean,
        "Negative electrode lithiation OCP [V]": graphite_ocp_lithiation,
        "Negative electrode delithiation OCP [V]": graphite_ocp_delithiation,

     
        #___________________________________________________________________________________________
        "Negative electrode porosity": 0.37,
        "Negative electrode active material volume fraction": 0.545, # Old initial Value: 0.52
        "Negative particle radius [m]": 3.5e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 3,
        "Negative electrode Bruggeman coefficient (electrode)": 2,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode OCP entropic change [V.K-1]": graphite_entropic_change,
        "Negative electrode exchange-current density [A.m-2]": graphite_exchange_current_density, # original value at 25°C 0.1932
        #"Negative electrode density [kg.m-3]": 1657.0,
        #"Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        #"Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        #"Negative electrode double-layer capacity [F.m-2]": 0.2,
        
        # Hysteresis parameters
        # Negative electrode (graphite)
        "Negative particle lithiation hysteresis decay rate": 100,
        "Negative particle delithiation hysteresis decay rate": 100,
        # starting from SOC=1: graphite was lithiating → h = -1 (lithiation branch)
        "Initial hysteresis state in negative electrode": -1,
        # Positive electrode (LFP)
        "Positive particle lithiation hysteresis decay rate": 100,
        "Positive particle delithiation hysteresis decay rate": 100,
        # starting from SOC=1: LFP was delithiating → h = +1 (delithiation branch)
        "Initial hysteresis state in positive electrode": 1,

        # Positive Electrode
        "Positive electrode conductivity [S.m-1]": 10,
        "Maximum concentration in positive electrode [mol.m-3]": 22806,
        "Positive electrode diffusivity [m2.s-1]": LFP_diffusivity,# original value at 25°C: 1.0386e-18,
        "Positive electrode OCP [V]": lfp_ocp_mean,
        "Positive electrode lithiation OCP [V]": lfp_ocp_lithiation,
        "Positive electrode delithiation OCP [V]": lfp_ocp_delithiation,
        "Positive electrode porosity": 0.4355,
        "Positive electrode active material volume fraction": 0.47954,
        "Positive particle radius [m]": 0.05e-06,
        "Positive electrode Bruggeman coefficient (electrode)": 1.5,
        "Positive electrode Bruggeman coefficient (electrolyte)": 2,
        "Positive electrode charge transfer coefficient": 0.5,
        # "Positive electrode particle hysteresis decay rate": 1,
        # "Positive electrode particle hysteresis switching factor": 1000,
        "Positive electrode OCP entropic change [V.K-1]": LFP_entropic_change,
        "Positive electrode exchange-current density [A.m-2]":0.033887*0.25,
        #"Positive electrode density [kg.m-3]": 2341.17,
        #"Positive electrode specific heat capacity [J.kg-1.K-1]": 1100.0,
        #"Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        #"Positive electrode double-layer capacity [F.m-2]": 0.2,


        # Separator
        "Separator porosity": 0.5,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        #"Separator density [kg.m-3]": 397.0,
        #"Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        #"Separator thermal conductivity [W.m-1.K-1]": 0.16,



        # Electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.38,
        "Thermodynamic factor": thermodynamic_factor_valoen,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Valeon2005,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Valeon2005,



        # Experimental Set-Up
        "Reference temperature [K]": 298.15,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 1.95,
        "Upper voltage cut-off [V]": 3.65,
        'Open-circuit voltage at 100% SOC [V]': 3.6,
        'Open-circuit voltage at 0% SOC [V]': 2.0,
        "Initial temperature [K]": 298.15,
        #"Heat transfer coefficient [W.m-2.K-1]": 10.0,




        # Electrode Balancing
        "Initial concentration in negative electrode [mol.m-3]": (0.0001 * 0 + 0.713 * 1) * 31370, 
        "Initial concentration in positive electrode [mol.m-3]": (0.85887 * 0 + 0.0015 * 1) * 22806,

        
        # citations
        "citations": ["Chen2020", "Lain2019", "Prada2013", "Nyman2008"],
    }


# %%
