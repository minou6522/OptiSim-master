from PyQt5.QtGui import QColor
import numpy as np
import numexpr as ne

class Layer:
    '''
    Class representing a layer with properties such as 'name', 'thickness', and 'nk'.
    '''
    def __init__(self, name,
                thickness=100, 
                criSource='constant',  
                criDBName = 'Si', 
                criFile = {'path': '', 'alpha': False, 'n': 2.0}, 
                criConstant = [1.0, 0.0]):
        '''
        Initializes the Layer object with various properties.
        
        Parameters:
        - name: Name of the layer
        - thickness: Thickness of the layer
        - criSource: Source of the refractive index ('constant', 'MaterialDB', 'file')
        - criDBName: Name in the material database
        - criFile: Dictionary containing file path and other properties for criSource 'file'
        - criConstant: Constant refractive index values [n, k]
        '''
        self.name = name
        self.parentName = name
        self.thickness = thickness
        self.thick = False
        self.criSource = criSource
        self.criFile = criFile.copy()
        self.criDBName = criDBName
        self.criConstant = criConstant

        self.drude_params = {'omega_p': 1.0, 'gamma': 0.1}
        def calculate_drude_permittivity(self, wavelength):
            omega = 2 * np.pi * 3e8 / wavelength  # Convert wavelength to angular frequency
            omega_p = self.drude_params['omega_p']
            gamma = self.drude_params['gamma']
            epsilon_inf = 1.0  # High-frequency permittivity (can be adjusted or made a parameter)
            epsilon = epsilon_inf - (omega_p**2 / (omega**2 + 1j * gamma * omega))
            return epsilon
        def update_dielectric_function(self, wavelength_range):
            self.dielectricFunction['wvl'] = wavelength_range
            epsilon = [self.calculate_drude_permittivity(wvl) for wvl in wavelength_range]
            self.dielectricFunction['n'] = np.real(np.sqrt(epsilon))
            self.dielectricFunction['k'] = np.imag(np.sqrt(epsilon))
        from scipy.optimize import minimize

        def fit_drude_parameters(self, observed_spectrum, wavelength_range):
            def error_function(params):
                self.drude_params['omega_p'], self.drude_params['gamma'] = params
                self.update_dielectric_function(wavelength_range)
                modeled_spectrum = self.calculate_reflection_spectrum()
                return np.sum((observed_spectrum - modeled_spectrum) ** 2)
    
            initial_guess = [self.drude_params['omega_p'], self.drude_params['gamma']]
            result = minimize(error_function, initial_guess)
            self.drude_params['omega_p'], self.drude_params['gamma'] = result.x

        # Grading properties for compositional grading within the layer
        self.criGrading = {
            'mode': 'constant', 
            'value': 0.0, 
            'files': [[0, 1, self.criFile['path']], [1, 2, self.criFile['path']]], 
            'xMoles': [],
            'Egs': [], 
            'n_idc': [], 
            'k_idc': []
        }
        
        # Dielectric function properties for modeling the layer's optical behavior
        self.dielectricFunction = {
            'e0': 0.0, 
            'oscillators': [{'name': 'Lorentz', 'values': [1, 3, 0.2], 'active': True}], 
            'spectral range': [0.5, 4.0], 
            'n': [], 
            'k': [], 
            'wvl': []
        }
        
        # Meshing properties for discretizing the layer
        self.mesh = {
            'meshing': 0, 
            'Points': 100, 
            'Dist': 1, 
            'refine': False
        }
        
        self.color = QColor(255, 255, 255)
        self.srough = False
        self.sroughThickness = 0
        self.sroughHazeR = 0.0
        self.sroughHazeT = 0.0
        
        self.x = None
        self.wavelength = None
        self.n = None
        self.k = None
        
        # Collection properties for charge collection in solar cells
        self.collection = {
            'source': 'from collection function', 
            'mode': 'constant', 'value': 1.0, 
            'SCRwidth': 300, 'diffLength': 1000, 'recVel': 1e7, 'SCRside': 0
        }
        
        self.makeXnodes()
        self.makeXcollection()
        self.makeXgrading()
    
    def makeXnodes(self):
        '''
        Creates the mesh nodes for the layer based on meshing settings.
        '''
        mode = self.mesh['meshing']
        number = self.mesh['Points']
        step = self.mesh['Dist']
        
        if mode == 2:  # Optimized meshing
            self.x = [0]
            x = 1
            while x < self.thickness / 2:
                self.x.append(x)
                x *= 1.1
            self.x2 = [self.thickness - i for i in self.x]
            self.x.extend(self.x2[::-1])
        elif mode == 1:  # Constant distance meshing
            self.x = np.arange(0, self.thickness, step)
        else:  # Fixed number of points
            self.x = np.linspace(0, self.thickness, number)
        
        self.x = np.array(self.x)
    
    def makeXcollection(self):
        '''
        Creates the charge collection profile based on collection settings.
        '''
        if self.collection['source'] == 'from collection function':
            if self.collection['mode'] == 'constant':
                fc = np.ones(len(self.x)) * self.collection['value']
            elif self.collection['mode'] == 'linear':
                fc = -((self.collection['value'][0] - self.collection['value'][1]) * self.x / self.thickness) + self.collection['value'][0]
            elif self.collection['mode'] == 'function':
                x = self.x
                dx = self.thickness
                fc = ne.evaluate(self.collection['value'])
            fc[fc > 1] = 1.0
            fc[fc < 0] = 0.0
            self.fc = np.array(fc)
        else:  # From diffusion length for constant difflength and constant field
            fc = np.zeros(len(self.x))
            W_scr = self.collection['SCRwidth']
            L = self.collection['diffLength']
            beta = self.collection.get('grading', 0.0)
            
            x = self.x
            W_abs = self.thickness
            S = self.collection['recVel']  # Surface recombination velocity [cm/s]
            D = 1.55  # Diffusion constant [cm²/s]
            kB = 8.617e-5  # Boltzmann constant [eV/K]
            T = 300  # Temperature [K]
            chi = 1e-6 * beta / (kB * T)  # Reduced field [1/nm]
            L_ = L / np.sqrt(1 + (chi * L / 2) ** 2)  # Adjusted diffusion length [nm]
            S_ = S + chi * D * 1e7 / 2  # Adjusted surface recombination velocity [cm/s]
            
            if self.collection['SCRside'] == 1:  # Bottom
                fc[self.x[::-1] <= W_scr] = 1
            else:  # Top
                fc = np.exp(chi * (x - W_scr) / 2) * (
                    np.cosh((W_abs - (x - W_scr)) / L_) +
                    1e-7 * S_ * L_ / D * np.sinh((W_abs - (x - W_scr)) / L_)
                ) / (np.cosh(W_abs / L_) + 1e-7 * (S_ * L_ / D) * np.sinh(W_abs / L_))
            
            fc[fc > 1] = 1.0
            fc[fc < 0] = 0.0
            self.fc = fc
    
    def makeXgrading(self):
        '''
        Creates the grading profile for the layer's composition.
        '''
        if self.criGrading['mode'] == 'constant':
            self.xMole = np.ones(len(self.x)) * self.criGrading['value']
        elif self.criGrading['mode'] == 'linear':
            self.xMole = -((self.criGrading['value'][0] - self.criGrading['value'][1]) * self.x / self.thickness) + self.criGrading['value'][0]
        elif self.criGrading['mode'] == 'function':
            x = self.x
            dx = self.thickness
            self.xMole = ne.evaluate(self.criGrading['value'])
        self.xMole[self.xMole > 1] = 1.0
        self.xMole[self.xMole < 0] = 0.0
