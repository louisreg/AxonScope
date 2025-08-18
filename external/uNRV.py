import math
import numpy as np
from neuron import h, load_mechanisms
import time 
from pathlib import Path
#pathlib.Path(__file__).parent.resolve()

#h.CVode.cache_efficient(True)

def is_empty_iterable(x):
    """ """
    if not np.iterable(x):
        return False
    if len(x) == 0:
        return True
    return False

def purge_neuron():
    """
    This function clears all the sections declared in h
    """
    for section in h.allsec():
        section = None

load_mechanisms( f"{Path(__file__).parent}/mods")
h.load_file("stdrun.hoc")

class Axon():

    def __init__(
        self,
        y=0,
        z=0,
        d=1,
        L=1000,
        dt=0.001,
        Nsec=100,
        Ra = 100,
        cm = 1,
        e_pas = -70,
        g_pas = 0.001,
        V_init = None,
        passif_cable = False,
        record_V_mem=True,
        record_I_mem=False,
        record_I_ions=False,
        record_particles=False,
        record_g_mem=False,
    ):
        
        self.Ra = Ra
        self.cm = cm
        self.e_pas = e_pas
        self.g_pas = g_pas
        self.y = y
        self.z = z
        self.d = d
        self.L = L
        self.dt = dt 
        self.Nsec = Nsec
        self.record_V_mem = record_V_mem
        self.record_I_mem = record_I_mem
        self.record_I_ions = record_I_ions
        self.record_particles = record_particles
        self.record_g_mem = record_g_mem
        self.passif_cable = passif_cable
        self.intra_current_stim = []
        self.intra_current_stim_positions = []
        self.intra_current_stim_starts = []
        self.intra_current_stim_durations = []
        self.intra_current_stim_amplitudes = []
        self.intra_voltage_stim = None
        self.intra_voltage_stim_position = []
        self.intra_voltage_stim_stimulus = None
        self.Nrec = 0
        if V_init is None:
            self.v_init = -70
        else:
            self.v_init = V_init

        self.__compute_axon_parameters()
        

    def __del__(self):
        pass
        #purge_neuron()

    def __clear_reclists(self):
        keys = list(self.__dict__.keys())
        for key in keys:
            if "reclist" in key:
                del self.__dict__[key]

    def __define_shape(self):
        """
        Define the shape of the axon after all sections creation
        """
        h.define_shape()

    def topology(self):
        """
        call the neuron topology function to plot the current topology on prompt
        """
        h.topology()


    def __compute_axon_parameters(self):
        """
        generate axon from parameters set by user
        """
        ## Handling v_init
        h.finitialize(self.v_init)  # initialize voltage state

        ## Handling temperature
        self.T = 37  # mamalian models
        self.model = "Rattay_Aberham"

        # create and connect
        self.unmyelinated_sections = [
             h.Section(name="U_axon[%d]" % i) for i in range(self.Nsec)
        ]
        for sec in self.unmyelinated_sections:
            # morphologic parameters
            sec.L = self.L / self.Nsec
            sec.diam = self.d
        for i in range(self.Nsec - 1):
            self.unmyelinated_sections[i + 1].connect(
                self.unmyelinated_sections[i], 1, 0
            )
        # implement neuron mechanisms
        self.__set_model()
        # define the geometry of the axon

        h.define_shape()
        # define the number of segments
        self.__set_Nseg()
        # get nodes positions
        self.__get_seg_positions()
        self.__get_rec_positions(self.Nrec)

    def __get_seg_positions(self):
        """
        get all the computation position of the axon. This corresponds to the positions of the segment
        without the duplicates of connected sections
        """
        x_offset = 0
        x = []
        for sec in self.unmyelinated_sections:
            for seg in sec.allseg():
                if is_empty_iterable(x):
                    x.append(seg.x * (sec.L) + x_offset)
                else:
                    x_seg = seg.x * (sec.L) + x_offset
                    if x_seg != x[-1]:
                        x.append(x_seg)
            x_offset += sec.L
        self.x = np.asarray(x)

    def __set_Nseg(self): 
        # set the number of segment for all declared sections
        self.Nseg  = 0
        for sec in self.unmyelinated_sections:
            sec.nseg = 1
            self.Nseg += 1


    def __get_rec_positions(self, Nrec):
        """
        get recording x-coordinates and relative positions, for internal use only

        Parameters
        ----------
        Nrec    : int
            number of points to record in the axon during the simulation, can be chosen between 3 and maximum the number of programmed segments + 2 (both sides)
            or will be by default set to the nearest value
        """
        self.rec_position_list = []
        for k in range(self.Nsec):
            self.rec_position_list.append([])
        if Nrec < 4 and Nrec != 0:
            self.Nrec = 3  # at least, 3 positions will be memorized along the axon (extrema and middle)
            self.rec_position_list[0].append(0)
            if (self.Nsec % 2) == 0:
                self.rec_position_list[self.Nsec / 2].append(0)
            else:
                self.rec_position_list[math.floor(self.Nsec / 2)].append(0.5)
            self.rec_position_list[-1].append(1)
            self.x_rec = np.array([0, self.L / 2, self.L])
        elif Nrec == 0 or Nrec > self.Nsec + self.Nseg:  # record on all nodes
            self.Nrec = self.Nsec + self.Nseg + 1
            for k in range(self.Nsec):
                for seg in self.unmyelinated_sections[k].allseg():
                    self.rec_position_list[k].append(seg.x)
                # delete last position as it will be a dupplicate with next section first seg
                del self.rec_position_list[k][-1]
            # for the last section only, add the max position
            self.rec_position_list[-1].append(1)
            self.x_rec = self.x
        else:
            self.Nrec = Nrec
            self.rec_position_list[0].append(0)
            remaining_recs = np.arange(1, self.Nrec - 1) * (self.Nsec / (self.Nrec - 1))
            for rec in remaining_recs:
                self.rec_position_list[math.floor(rec)].append(rec - math.floor(rec))
            self.rec_position_list[-1].append(1)
            self.x_rec = np.linspace(0, self.L, num=self.Nrec, endpoint=True)

    def __set_model(self):
        """
        Adds the passive, Hodgking Huxley and extracellular mechanisms,
        set values to to one given by user at the initialisation or default ones. For internal use only.
        """
        for sec in self.unmyelinated_sections:
            # insert mechanisms
            
            sec.insert("pas")
            #sec.insert("extracellular") #@Todo: remove if used 
            #sec.xg[0] = 1e10  # short circuit, no myelin
            #sec.xc[0] = 0  # short circuit, no myelin
            ## Except for HH, the following code is directly take from modelDB: https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=266498#tabs-1
            ## Pelot N (2020) Excitation Properties of Computational Models of Unmyelinated Peripheral Axons J Neurophysiology

            #create passive cable for comparison
            sec.Ra = self.Ra 
            sec.cm = self.cm 
            sec.e_pas = self.e_pas 
            sec.g_pas = self.g_pas 
            sec.v  = self.v_init

            if not (self.passif_cable):
                sec.insert(
                    "RattayAberham"
                )  # Model adjusted for a resting potential of -70mV instead of 0 (subtract Vrest from each reversal potential)
                sec.ena = 45
                sec.ek = -82





    ###############################
    ## Intracellular stimulation ##
    ###############################
    def insert_I_Clamp(self, position, t_start, duration, amplitude):
        """
        Insert a I clamp stimulation

        Parameters
        ----------
        position    : float
            relative position over the axon
        t_start     : float
            starting time, in ms
        duration    : float
            duration of the pulse, in ms
        amplitude   : float
            amplitude of the pulse (nA)
        """
        # adapt position to the number of sections
        portion_length = 1.0 / self.Nsec
        stim_sec = int(math.floor(position / portion_length))
        stim_pos = (position / portion_length) - math.floor(position / portion_length)
        # add the stimulation to the axon
        self.intra_current_stim.append(
            h.IClamp(stim_pos, sec=self.unmyelinated_sections[stim_sec])
        )
        # modify the stimulation parameters
        self.intra_current_stim[-1].delay = t_start
        self.intra_current_stim[-1].dur = duration
        self.intra_current_stim[-1].amp = amplitude
        # save the stimulation parameter for results
        self.intra_current_stim_positions.append(position * self.L)
        self.intra_current_stim_starts.append(t_start)
        self.intra_current_stim_durations.append(duration)
        self.intra_current_stim_amplitudes.append(amplitude)


    ##############################
    ## Result recording methods ##
    ##############################
    def __set_recorders_with_key(self, *args):
        """
        To automate the methods set_recorder. For internal use only.
        Parameters
        ----------
        *args    : list(tuple)
            list of tuple containing a rec list to set and the corresonding key to access
            NB: keys should be str such as "_ref_xxx_yyy" where xxx is the variable to access
            and yyy the .mod file suffix if the variable is in one
        """
        for k in range(self.Nsec):
            for pos in self.rec_position_list[k]:
                for t in args:
                    key = t[1]
                    # print(dir(self.unmyelinated_sections[k](pos)))
                    # print(key, getattr(self.unmyelinated_sections[k](pos),key))
                    rec = h.Vector().record(
                        getattr(self.unmyelinated_sections[k](pos), key),
                        sec=self.unmyelinated_sections[k],
                    )
                    t[0].append(rec)

    def __get_var_from_mod(self, key):
        """
        return a column with value in every recording point of a constant from a mod. For internal use only.
        """
        val = np.zeros((len(self.x_rec)))
        i = 0
        for k in range(self.Nsec):
            for pos in self.rec_position_list[k]:
                # print(getattr(self.unmyelinated_sections[k](pos), key)[0])
                val[i] = getattr(self.unmyelinated_sections[k](pos), key)[0]
                i += 1
        return val

    def __get_recorders_from_list(self, reclist):
        """
        Convert reclist in np.array To automate methods set_recorder. For internal use only.
        Parameters
        ----------
        reclist     : h.List
            List in witch the reccorders are saved

        Returns
        -------
        val         : np.array
            array of every recorded value for all rec point and time
        """
        dim = (self.Nrec, self.t_len)
        val = np.zeros(dim)
        for k in range(dim[0]):
            val[k, :] = np.asarray(reclist[k])
        return val

    def set_membrane_voltage_recorders(self):
        """
        Prepare the membrane voltage recording. For internal use only.
        """
        self.vreclist = h.List()
        self.__set_recorders_with_key((self.vreclist, "_ref_v"))

    def get_membrane_voltage(self):
        """
        get the membrane voltage at the end of simulation. For internal use only.
        """
        return self.__get_recorders_from_list(self.vreclist)

    def set_membrane_current_recorders(self):
        """
        Prepare the membrane current recording. For internal use only.
        """
        self.ireclist = h.List()
        self.__set_recorders_with_key((self.ireclist, "_ref_i_membrane"))

    def get_membrane_current(self):
        """
        get the membrane current at the end of simulation. For internal use only.
        """
        return self.__get_recorders_from_list(self.ireclist)

    def set_ionic_current_recorders(self):
        """
        Prepare the ionic currents recording. For internal use only.
        """

        self.i_na_reclist = h.List()
        self.i_k_reclist = h.List()
        self.i_l_reclist = h.List()
        self.__set_recorders_with_key(
            (self.i_na_reclist, "_ref_nai"),
            (self.i_k_reclist, "_ref_ki"),
            (self.i_l_reclist, "_ref_i_pas"),
        )


    def get_ionic_current(self):
        """
        get the ionic currents at the end of simulation. For internal use only.
        """
        results = []
        results += [self.__get_recorders_from_list(self.i_na_reclist)]
        results += [self.__get_recorders_from_list(self.i_k_reclist)]
        results += [self.__get_recorders_from_list(self.i_l_reclist)]

        return results

    def set_conductance_recorders(self):
        """
        Prepare the membrane conductance recording. For internal use only.
        """
        self.g_na_reclist = h.List()
        self.g_k_reclist = h.List()
        self.g_l_reclist = h.List()

        self.__set_recorders_with_key(
            (self.g_na_reclist, "_ref_gna_RattayAberham"),
            (self.g_k_reclist, "_ref_gk_RattayAberham"),
            (self.g_l_reclist, "_ref_gl_RattayAberham"),
        )

    def get_membrane_conductance(self):
        """
        get the membrane voltage at the end of simulation. For internal use only.
        NB: [S/cm^{2}] (see Neuron unit)
        """
        return sum(self.get_ionic_conductance())

    def get_ionic_conductance(self):
        """
        get the membrane conductance at the end of simulation. For internal use only.
        NB: [S/cm^{2}] (see Neuron unit)
        """
        results = []
        #results += [self.__get_recorders_from_list(self.g_na_reclist)]
        #results += [self.__get_recorders_from_list(self.g_k_reclist)]
        results += [self.__get_recorders_from_list(self.g_l_reclist)]
        return results

    def get_membrane_capacitance(self):
        """
        get the membrane capacitance
        NB: [uF/cm^{2}] (see Neuron unit)
        """
        return self.__get_var_from_mod("_ref_cm")

    def set_particules_values_recorders(self):
        """
        Prepare the particule value recording. For internal use only.
        """

        self.hhmreclist = h.List()
        self.hhnreclist = h.List()
        self.hhhreclist = h.List()
        self.__set_recorders_with_key(
            (self.hhmreclist, "_ref_m_RattayAberham"),
            (self.hhnreclist, "_ref_n_RattayAberham"),
            (self.hhhreclist, "_ref_h_RattayAberham"),
        )


    def get_particles_values(self):
        """
        get the particules values at the end of simulation. For internal use only.
        """
        results = []
        results += [self.__get_recorders_from_list(self.hhmreclist)]
        results += [self.__get_recorders_from_list(self.hhnreclist)]
        results += [self.__get_recorders_from_list(self.hhhreclist)]
        return results
    
    def __get_time_vector(self):
        """
        internal use: get the time vector and stor its length for further use

        Returns
        -------
        t   : np.array
            time vector of the previous simulation, numpy array
        """
        t = np.array(self.timeVector)
        self.t_len = len(t)
        return t

    def simulate(
        self,
        t_sim,
    ):
        """
        Simulates the axon using neuron framework

        Parameters
        ----------
        t_sim               : float
            total simulation time (ms), by default 20 ms
        self.record_V_mem        : bool
            if true, the membrane voltage is recorded, set to True by default
                see unmyelinated/myelinated to see where recording occur
                results stored with the key "V_mem"
        self.record_I_mem        : bool
            if true, the membrane current is recorded, set to False by default
        self.record_I_ions       : bool
            if true, the ionic currents are recorded, set to False by default
        record_particules   : bool
            if true, the marticule states are recorded, set to False by default
        self.loaded_footprints           :dict or bool
            Dictionnary composed of extracellular footprint array, the keys are int value
            of the corresponding electrode ID, if None, footprints calculated during the simulation,
            set to None by default

        Returns
        -------
        axon_sim    : dictionnary
            all informations on neuron, segment position and all simulation results
        """
        self.t_sim = t_sim
        axon_sim = {}
        axon_sim["diameter"] = self.d
        axon_sim["tstop"] = self.t_sim
        #axon_sim.update(self.__add_extrastim_to_res())
        
        # set recorders arrays - KEEP THIS CODE BEFORE INITIALISATION
        self.timeVector = h.Vector().record(h._ref_t)

        if self.record_V_mem:
            self.set_membrane_voltage_recorders()
        if self.record_I_mem:
            self.set_membrane_current_recorders()
        if self.record_I_ions:
            self.set_ionic_current_recorders()
        if self.record_particles:
            self.set_particules_values_recorders()

        ## initialisation and parameters for neuron - KEEP THIS CODE JUST BEFORE SIMULATION
        h.tstop = t_sim
        h.celsius = self.T  # set temperature in celsius
        h.finitialize(self.v_init)  # initialize voltage state
        h.v_init = self.v_init

        h.dt = self.dt  # set time step (ms)

        start_time = time.time()
        h.frecord_init()
        ###########################################
        #### THIS IS WHERE SIMULATION IS HANDLED ##
        ###########################################

        if self.intra_voltage_stim is not None:
            # init if first point is at 0
            if self.intra_voltage_stim_stimulus.t[0] == 0:
                self.intra_voltage_stim.amp[0] = self.intra_voltage_stim_stimulus.s[
                    0
                ]
            # run simulation with a loop on the voltage clamp times
            for i in range(1, len(self.intra_voltage_stim_stimulus.t)):
                t_step = min(self.intra_voltage_stim_stimulus.t[i], t_sim)
                # run simulation
                h.continuerun(t_step)
                # apply new voltage clamp value
                self.intra_voltage_stim.amp[0] = self.intra_voltage_stim_stimulus.s[
                    i
                ]
            # finish simulation if needed
            if h.t < t_sim:
                h.continuerun(h.tstop)
        else:
            h.continuerun(h.tstop)
        ###########################################
        ###########################################
        ###########################################
        self.sim_time = time.time() - start_time
        # simulation done, store results
        axon_sim["Simulation_state"] = "Successful"
        axon_sim["sim_time"] = self.sim_time
        axon_sim["t"] = self.__get_time_vector()
        axon_sim["x_rec"] = self.x_rec

        if self.record_V_mem:
            axon_sim["V_mem"] = self.get_membrane_voltage()
        if self.record_I_mem:
            axon_sim["I_mem"] = self.get_membrane_current()

        if self.record_I_ions:
            I_na_ax, I_k_ax, I_l_ax = self.get_ionic_current()
            axon_sim["I_na"] = I_na_ax
            axon_sim["I_k"] = I_k_ax
            axon_sim["I_l"] = I_l_ax

        if self.record_particles:
            m_ax, n_ax, h_ax = self.get_particles_values()
            axon_sim["m"] = m_ax
            axon_sim["n"] = n_ax
            axon_sim["h"] = h_ax


        self.__clear_reclists()
        return axon_sim