import numpy as np                        # need everywhere
import matplotlib.pyplot as plt           # need everywhere
import matplotlib.cm as cm                # color gradation
from matplotlib.gridspec import GridSpec  # for summary plot
from mpl_toolkits.mplot3d import axes3d   # for 3D object of super-ellipse SWS

import cv2                                # To find the contour of each slice


import scipy.special as special           # for modeling klopfenstein
import scipy.integrate as integrate       # for moleling klopfenstein
import scipy.ndimage as ndi               # To what?? maybe fine to remove
from scipy.special import gamma           # for modeling klopfenstein
from scipy.optimize import curve_fit      # To fit the contour of each slice

from glob import glob                     # get filename
from natsort import natsorted             # sort files
from tqdm.notebook import tqdm            # progress bar (for jupyter, if you use local .py you may need to remove ".notebook")


import binascii
import ipywidgets as widgets





#import matplotlib.ticker as ticker        
#import vk4_analysis as vk

class Gen_SuperEllipse:
    """
    Class for the tutorial of super sllipse
    1. Contour_SuperEllipse
        Make a single super ellipse contour given parameters
        parameters:
            rx: radius in x (int, float)
            ry: radius in y (int, float)
            nx: index of super ellipse in x (int, float)
            ny: index of super ellipse in y (int, float)
        return 
            x, y in parametric display (1-D numpy array)

    2. Contour_SuperEllipse_arr
        Make superllipse contours given parameter arrays
        parameters:
            rx_arr: radius in x (1-D numpy array)
            ry_arr: radius in y (1-D numpy array)
            nx_arr: index of super ellipse in x (1-D numpy array)
            ny_arr: index of super ellipse in y (1-D numpy array)
        return
            x_arr, y_arr in parametric display (2-D numpy array)

    3. Plot_Contour_SuperEllipse
        Plot contour super ellise given array
        parameters:
            x_arr: x array in parametric display (2-D numpy array)
            y_arr: y array in parametric display (2-D numpy array)
            nx_arr: index of super ellipse in x (1-D numpy array)
            ny_arr: index of super ellipse in y (1-D numpy array)
            sname: name for saving the figure (string), saved in ./Figure directry
        return:     
    """
    def __init__(self):
        self.dpoint = 1001
        self.theta = np.linspace(0,2*np.pi,self.dpoint)
        
    def Contour_SuperEllipse(self,rx,ry,nx,ny):
        """
        Contour_SuperEllipse
            Make a single super ellipse contour given parameters
            parameters:
                rx: radius in x (int, float)
                ry: radius in y (int, float)
                nx: index of super ellipse in x (int, float)
                ny: index of super ellipse in y (int, float)
            return 
                x, y in parametric display (1-D numpy array)
        """
        x = rx * np.sign(np.cos(self.theta))*abs(np.cos(self.theta))**(2/nx) # rx (array to make contour)
        y = ry * np.sign(np.sin(self.theta))*abs(np.sin(self.theta))**(2/ny) # ry (array to make contour)
        return x,y
    
    def Contour_SuperEllipse_arr(self,rx_arr,ry_arr,nx_arr,ny_arr):
        """
        Contour_SuperEllipse_arr
            Make superllipse contours given parameter arrays
            parameters:
                rx_arr: radius in x (1-D numpy array)
                ry_arr: radius in y (1-D numpy array)
                nx_arr: index of super ellipse in x (1-D numpy array)
                ny_arr: index of super ellipse in y (1-D numpy array)
            return
                x_arr, y_arr in parametric display (2-D numpy array)
        """
        x_arr = np.zeros([len(rx_arr),self.dpoint])
        y_arr = np.zeros([len(rx_arr),self.dpoint])
        for i in range(0,len(rx_arr)):
            x = rx_arr[i] * np.sign(np.cos(self.theta))*abs(np.cos(self.theta))**(2/nx_arr[i]) # rx (array to make contour)
            y = ry_arr[i] * np.sign(np.sin(self.theta))*abs(np.sin(self.theta))**(2/ny_arr[i]) # ry (array to make contour)
            x_arr[i] = x
            y_arr[i] = y
        return x_arr,y_arr
    
    def Plot_Contour_SuperEllipse(self,x_arr,y_arr,nx_arr,ny_arr,sname):
        """
        Plot_Contour_SuperEllipse
            Plot contour super ellise given array
            parameters:
                x_arr: x array in parametric display (2-D numpy array)
                y_arr: y array in parametric display (2-D numpy array)
                nx_arr: index of super ellipse in x (1-D numpy array)
                ny_arr: index of super ellipse in y (1-D numpy array)
                sname: name for saving the figure (string), saved in ./Figure directry
            return:   
        """
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)
        
        for ai in range(0,len(x_arr)):
            ax.plot(x_arr[ai],y_arr[ai],color = cm.jet(ai/len(x_arr)),label = '$n_x=%s$'%nx_arr[ai]+', $n_y=%s$'%ny_arr[ai])
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('$x$',fontsize = 12)
        ax.set_ylabel('$y$',fontsize = 12)
        ax.legend(bbox_to_anchor = (1.0, 1., 0.0, 0.0))
        fig.tight_layout()
        plt.savefig('./Figure/'+sname)
        plt.show()
        plt.close()
        
        
class Gen_SuperEllipse_z:
    """
    class for the tutorial of super sllipse including z dependence
    1. SuperEllipse_zdep
            Make contour of super ellipse for each depth with z_num
            parameters:
                z_num: number of depth (int)
                rx_arr: radius in x (1-D numpy array)
                ry_arr: radius in y (1-D numpy array)
                nx_arr: index of super ellipse in x (1-D numpy array)
                ny_arr: index of super ellipse in y (1-D numpy array)
                p: pitch of the structure (int, float, in [mm])
            return: x_arr, y_arr in parametric display (2-D numpy array)
    2.Plot_Contour_SuperEllipse_z
            Plot contour super ellise, radius and index of super ellipse given arrays
            parameters:
                z_arr: depth array (1-D numpy array)
                x_arr: x array in parametric display (2-D numpy array)
                y_arr: y array in parametric display (2-D numpy array)
                nx_arr: index of super ellipse in x (1-D numpy array)
                ny_arr: index of super ellipse in y (1-D numpy array)
                sname: name for saving the figure (string), saved in ./Figure directry
            return: 
    """
    def __init__(self):
        self.dpoint = 1001
        self.theta = np.linspace(0,2*np.pi,self.dpoint)
        
    def SuperEllipse_zdep(self,z_num,rx_arr,ry_arr,nx_arr,ny_arr,p):
        """
        SuperEllipse_zdep
            Make contour of super ellipse for each depth with z_num
            parameters:
                z_num: number of depth (int)
                rx_arr: radius in x (1-D numpy array)
                ry_arr: radius in y (1-D numpy array)
                nx_arr: index of super ellipse in x (1-D numpy array)
                ny_arr: index of super ellipse in y (1-D numpy array)
                p: pitch of the structure (int, float, in [mm])
            return: x_arr, y_arr in parametric display (2-D numpy array)
        """
        x_arr = np.zeros([z_num,self.dpoint])
        y_arr = np.zeros([z_num,self.dpoint])
        for i in range(0,z_num):
            x = rx_arr[i] * np.sign(np.cos(self.theta))*abs(np.cos(self.theta))**(2/nx_arr[i]) # rx (array to make contour)
            y = ry_arr[i] * np.sign(np.sin(self.theta))*abs(np.sin(self.theta))**(2/ny_arr[i]) # ry (array to make contour)
            x_arr[i] = x
            y_arr[i] = y
            
            x_arr[i,np.where(x_arr[i] >=  p/2)] = p/2
            x_arr[i,np.where(x_arr[i] <= -p/2)] = -p/2
            y_arr[i,np.where(y_arr[i] >=  p/2)] = p/2
            y_arr[i,np.where(y_arr[i] <= -p/2)] = -p/2
        return x_arr,y_arr
    
    def Plot_Contour_SuperEllipse_z(self,z_arr,rx_arr,ry_arr,x_arr,y_arr,nx_arr,ny_arr,sname):
        """
        Plot_Contour_SuperEllipse_z
            Plot contour super ellise, radius and index of super ellipse given arrays
            parameters:
                z_arr: depth array (1-D numpy array)
                rx_arr: radius in x (1-D numpy array)
                ry_arr: radius in y (1-D numpy array)
                x_arr: x array in parametric display (2-D numpy array)
                y_arr: y array in parametric display (2-D numpy array)
                nx_arr: index of super ellipse in x (1-D numpy array)
                ny_arr: index of super ellipse in y (1-D numpy array)
                sname: name for saving the figure (string), saved in ./Figure directry
            return:   
        """
        fig = plt.figure(figsize = (15,8))
        gs = GridSpec(2,5)

        ss1 = gs.new_subplotspec((0,0),rowspan = 2,colspan = 2)  # contour
        ss2 = gs.new_subplotspec((0,2),colspan = 3)              # radius (taper)
        ss3 = gs.new_subplotspec((1,2),colspan = 3)              # index (taper)
        
        ax1 = plt.subplot(ss1) # contour
        ax2 = plt.subplot(ss2) # radius (taper)
        ax3 = plt.subplot(ss3) # index (taper)
        
        # =========================================
        # Contour-like plot
        # - - - - - - - - - - - - - - - - - - - - -       
        x_arr_r = x_arr[::-1]
        y_arr_r = y_arr[::-1]
        for ai in range(0,len(x_arr)):
            ax1.plot(x_arr_r[ai],y_arr_r[ai],color = cm.jet(ai/len(x_arr)),label = '$n_x=%s$'%nx_arr[ai]+', $n_y=%s$'%ny_arr[ai])
        ax1.set_aspect('equal')
        ax1.grid(True)
        ax1.set_xlabel('$x$ [mm]',fontsize = 12)
        ax1.set_ylabel('$y$ [mm]',fontsize = 12)
        # - - - - - - - - - - - - - - - - - - - - - 
        # =========================================
        
        # =========================================
        # Radius [mm]
        # - - - - - - - - - - - - - - - - - - - - -          
        ax2.plot(z_arr,rx_arr,'b',label = '$r_x$')
        ax2.plot(z_arr,ry_arr,'r',label = '$r_y$')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlabel('Depth [mm]',fontsize =12)
        ax2.set_ylabel('Radius [mm]',fontsize = 12)
        # - - - - - - - - - - - - - - - - - - - - - 
        # =========================================
        
        # =========================================
        # Index of Super ellipse
        # - - - - - - - - - - - - - - - - - - - - -          
        ax3.plot(z_arr,nx_arr,'b',label = '$r_x$')
        ax3.plot(z_arr,ny_arr,'r',label = '$r_y$')
        ax3.legend()
        ax3.grid(True)
        ax3.set_xlabel('Depth [mm]',fontsize =12)
        ax3.set_ylabel('Index of SE',fontsize = 12)
        # - - - - - - - - - - - - - - - - - - - - - 
        # =========================================
        
        fig.tight_layout()
        plt.savefig('./Figure/'+sname)
        plt.show()
        plt.close()  
        


class Design_SWS:
    def __init__(self):
        # ==========================
        # Constants
        # - - - - - - - - - - - - - - -
        self.mu = 1.25663706e-06  # vacuum permeability [m kg s-2 A-2]
        self.ep0 = 8.8542e-12     # vacuum permitivity [m-3 kg-1 s4 A2]
        self.c = 2.9979e+08       # speed of light [m/s]
        self.pi = np.pi           # pi
        # - - - - - - - - - - - - - - -
        # ==========================
        
    def Klopfenstein(self,h,num,ni,ns,Gamma):
        '''
        * Def name
            Klopfenstein
        * Description
            Calculate Klofenstein index profile
            Based on two papers:
                1: Klopfenstein(1956):https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4051841
                2: Grann et al(1995): https://opg.optica.org/view_article.cfm?pdfKey=fbf947c3-8dd3-440b-bfe51c58d6e98f89_33114
        * input parameters
            - h:   height of the taper along z axis [arbitral unit]
            - num: number of layer (as same as resolution of depth of SWS)
            - ni:  refractive index of air, so 1.0
            - ns:  refractive index of substrate
            - Gamma: important parameter to control the trade-off between the ripple in the opera- tion bands and bandwidth

        * return
            - n_arr: Klopfenstein index profile
            - z_arr: array of depth (height)
            - z_space: array of thickness each layer
        '''
        z_arr = np.linspace(0,h,num)            # define array of depth (height)
        n_arr = np.zeros(num)                   # set array of Klopfenstein index profile
        z_space = np.ones(len(z_arr))*(h/num)   # calculate array of thickness each layer
        rho_0 = 0.5*np.log(ns/ni)               # calcualte rho_0(see Klopfenstein(1956))
        A = np.arccosh(rho_0/Gamma)             # calculate A(see Klopfenstein(1956))
        
        # Calculate Klopfenstein index profile 
        # (see Eq.12 in Klopfenstein(1956), and Eq.6 in Grann et al(1995))
        for i in range(0,num):
            x_i = 2.*z_arr[i]/h-1.
            phi_int = integrate.quad(lambda y: special.i1(A*np.sqrt(1.-y**2))/(A*np.sqrt(1.-y**2)), 0, x_i)  
            n_arr[i] = np.sqrt(ni*ns) * np.exp(Gamma * A**2 * phi_int[0])
        return n_arr,z_arr,z_space
    
    def fit_oblique_basic_multilayer_r_t_incloss(self, n, losstan, d, freq_in, angle_i, incpol):
        '''
        * Def name
            fit_oblique_basic_multilayer_r_t_incloss
        * Description
            Calculate coefficient of reflectance and transmittance based on Transfer matrix method, made by Tomo Matsumura
            (modified with this class by RTakaku)
        * input parameters
            - n:  refractive index of substrate
            - losstan: loss tangent of substrate
            - d: thickness [m] 
            - freq_in: input frequency [Hz]
            - angle_i: incident angle [rad]
            - incpol: 1 for s-state, E field perpendicular to the plane of incidnet, -1 for P-state, E in the plane of incident

        * return
            - output ([0]: freq, [1]: coeff of reflectance, [2]: coeff of transmittance (complex numpy array))
        '''
        num=len(d) #; the number of layer not including two ends
        const = np.sqrt((self.ep0)/(4.*self.pi*1e-7)) #SI unit sqrt(dielectric const/permiability)

        # ;-----------------------------------------------------------------------------------
        # ; angle of refraction
        angle = np.zeros(num+2)          # ; angle[0]=incident angle
        angle[0] = angle_i
        for i in range(0,num+1): angle[i+1] = np.arcsin(np.sin(angle[i])*n[i]/n[i+1])

        # ;-----------------------------------------------------------------------------------
        # ; define the frequency span
        l = len(freq_in)
        output = np.zeros((3,l),'complex') # output = dcomplexarr(3,l)

        # ;-----------------------------------------------------------------------------------
        # ; define the effective thickness of each layer
        h = np.zeros(num,'complex')
        n_comparr = np.zeros(len(n),'complex')
        n_comparr[0] = complex(n[0], -0.5*n[0]*losstan[0])
        n_comparr[num+1] = complex(n[num+1], -0.5*n[num+1]*losstan[num+1])

        # ;-----------------------------------------------------------------------------------
        # ; for loop for various thickness of air gap between each layer
        for j in range(0,l):
            for i in range(0,num): 
                n_comparr[i+1] = complex(n[i+1], -0.5*n[i+1]*losstan[i+1])
                h[i] = n_comparr[i+1]*d[i]*np.cos(angle[i+1]) # ;effective thickness of 1st layer

            freq = freq_in[j]
            k = 2.*self.pi*freq/self.c

            # ;===========================================
            # ; Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side
            Y = np.zeros(num+2,'complex')
            for i in range(0,num+2):
                if (incpol == 1):
                    Y[i] = const*n_comparr[i]*np.cos(angle[i])
                    cc = 1.
                if (incpol == -1):
                    Y[i] = const*n_comparr[i]/np.cos(angle[i])
                    cc = np.cos(angle[num+1])/np.cos(angle[0])

            # ;===========================================
            # ; define matrix for single layer
            m = np.identity((2),'complex')    # ; net matrix
            me = np.zeros((2,2),'complex') # ; me[0]=1st layer, ...
            for i in range(0,num):
                me[0,0] = complex(np.cos(k*h[i]), 0.)
                me[1,0] = complex(0., np.sin(k*h[i])/Y[i+1])
                me[0,1] = complex(0., np.sin(k*h[i])*Y[i+1])
                me[1,1] = complex(np.cos(k*h[i]), 0.)
                m = np.dot(m,me)

            r = (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]-m[0,1]*cc-Y[num+1]*m[1,1]) / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
            t = 2.*Y[0] / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])

            output[0,j] = freq+0.j #; unit of [Hz]
            output[1,j] = r
            output[2,j] = t

        return output
    
    def Brauer_emt_anti_symmetric(self,freq,n1,n2,f1,f2,p1,p2):
        '''
        * Def name
            Brauer_emt_anti_symmetric
        * Description
            Calculate effective refractive index given area fraction
            Based on Brauer(1994): https://opg.optica.org/view_article.cfm?pdfKey=92654043-e0c1-4520-98b312f00908c33d_42237
        * input parameters
            - freq:   inpu frequency [Hz]
            - n1: refractive index of air, so 1.0
            - n2: refractive index of substrate
            - f1: area fraction along x axis
            - f2: area fraction along y axis
            - p1: pitch x
            - p2: pitch y

        * return
            - n_: 0th ordered effective refractive index
            - neff: 2nd ordered effective refractive index
        '''
        
        lamda = self.c/freq   # wavelength
        f = (f1+f2)/2.        # average fraction in x and y

        e1 = n1**2.*self.ep0  # refractive index --> permitivity (air) 
        e2 = n2**2.*self.ep0  # refractive index --> permitivity (substrate)

        ell_0 = (1.0 - f1)*e1+f1*e2    # Eq.1
        els_0 = 1./((1.-f2)/e1+f2/e2)  # Eq.2

        ell_2 = ell_0*(1.+(np.pi**2/3.)*(p1/lamda)**2.*f1**2*(1.-f1)**2.*((e2-e1)**2./(self.ep0*ell_0)))                    # Eq.3
        els_2 = els_0*(1.0+(np.pi)**2/3.0*(p2/lamda)**2*f2**2*(1.-f2)**2.*((e2-e1)**2.)*ell_0/self.ep0*(els_0/(e2*e1))**2.) # Eq.4
        
        e_2nd_up = (1.0 - f1)*e1 + f1*els_2           # Eq.6
        e_2nd_down = 1./((1.0 - f2)/e1 + f2/ell_2)    # Eq.7

        n_=(1-f**2)*n1+f**2*n2                        # Eq.5
        n__2nd_up = np.sqrt(e_2nd_up/(self.ep0))      # permitivity --> refractive index (up)
        n__2nd_down = np.sqrt(e_2nd_down/(self.ep0))  # permitivity --> refractive index (down)

        neff = 0.2*(n_+2.0*n__2nd_up+2.0*n__2nd_down) # Eq.8
        return n_*np.ones(len(neff)),neff
    
    def Calculate_transmittance_include_Klopfenstein_profile_on_both_sides(self,freq, input_n,input_losstan,input_d, h, z_res, Gamma,Plot_klop_profile = False):
        # ==========================
        # Calculate Klopfenstein index profile and its transmission
        # - - - - - - - - - - - - - - -
        n_arr, z_klop, d_klop = self.Klopfenstein(h,z_res,1.0,input_n,Gamma) # n, z, thickness each layer
        
        if Plot_klop_profile == True:
            fig = plt.figure(figsize = (6,4))
            ax = fig.add_subplot(111)
            ax.plot(n_arr,z_klop[::-1]*1e+3, 'r',label = r'$\Gamma_m = %.3f$'%Gamma)
            ax.grid()
            ax.set_xlabel('Effective refractive index',fontsize = 15)
            ax.set_ylabel('Height [mm]', fontsize = 15)
            ax.legend()
            fig.tight_layout()
            plt.show()
            plt.close()
        n_klop = np.concatenate((np.array([1.0]),n_arr))                       # just add air layer (incident environment) to n_arr, which is exactly Klopfenstein index profile
        thickness_for_klop = np.concatenate((d_klop,np.array([input_d]),d_klop))   # thickness array ( Klopfesntein array (N of layer) + substrate + Klopfesntein array (N of layer) )   
        index_for_klop = np.concatenate((n_klop,np.array([input_n]),n_klop[::-1])) # refractive index array ( air + Klopfesntein array (N of layer) + substrate + Klopfesntein array (N of layer) + air)
        losstan_for_klop = np.concatenate((np.ones(len(n_klop))*input_losstan,np.array([input_losstan]),np.ones(len(n_klop))*input_losstan)) 
                                                                                   # loss tangent array ( air + Klopfesntein array (N of layer) + substrate + Klopfesntein array (N of layer) + air)

        trans_klop = abs(self.fit_oblique_basic_multilayer_r_t_incloss(index_for_klop, losstan_for_klop, thickness_for_klop,freq,angle_i=0, incpol=1)[2])**2 
                                                                                   # This indicates |trans. coeff|**2
        
        # Now, we have n_klop(Klopfenstein index profile), and trans_klop (Transmission based on Klofenstein ARC)
        # - - - - - - - - - - - - - - -
        # ==========================
        return n_arr, z_klop, trans_klop
    
    def Find_bandwidth(self,vc,fc):
        diff = vc*fc/2
        
        band_low = vc - diff
        band_high = vc + diff
        
        res = np.transpose(np.array([band_low,band_high]))
        
        return res
    
    def Calculate_band_average(self,freq,trans,band):
        band_num = len(band)
        ave_arr = np.empty(band_num)
        
        for i in range(0,band_num):
            freq_ind = np.where((freq>=band[i][0])&(freq<=band[i][1]))
            ave_arr[i] = np.average(trans[freq_ind])
        return ave_arr
    
    def Plot_transmittance_EMT(self,freq,trans,trans_ave,band,center_freq):
        fig = plt.figure(figsize = (6,4))
        ax = fig.add_subplot(111)
        
        ax.plot(freq*1e-9,trans, 'r')
        
        ax.set_xlabel('Frequency [GHz]',fontsize = 15)
        ax.set_ylabel('Transmittance',fontsize = 15)
        
        ax.grid(True)
        y_minimum = 0.9
        ax.set_ylim(y_minimum,1.02)
        
        for i in range(0,len(center_freq)):
            ax.fill_between(band[i]*1e-9,y1 = np.ones(2)*0, y2 = np.ones(2)*3, color = 'r', alpha = 0.2)
            ax.text(center_freq[i]*1e-9,y_minimum+0.01,r'$T_{ave}= $%.3f'%trans_ave[i],ha = 'center',va = 'top')
        
        fig.tight_layout()
        plt.show()
        plt.close()
        
    def Find_width(self,arr, n0, ns, vc, p):
        area_frac_arr = np.linspace(0,1.0,10001)
        n_eff_arr = self.Brauer_emt_anti_symmetric(vc,n0,ns,area_frac_arr,area_frac_arr,p,p)
        
        w = np.zeros(len(arr))
        for i in range(0,len(w)):
            w[i] = p*area_frac_arr[np.argmin(abs(arr[i] - n_eff_arr[1]))]
        return w
    
    def Plot_cross_section(self,width_arr,klop_d,p):
        width_for_plot = np.concatenate((-width_arr[::-1]/2-p,width_arr/2-p,-width_arr[::-1]/2,width_arr/2,-width_arr[::-1]/2+p,width_arr/2+p))*1e+3
        depth_for_plot = np.concatenate((klop_d[::-1],klop_d,klop_d[::-1],klop_d,klop_d[::-1],klop_d))*1e+3
        
        
        fig = plt.figure(figsize = (6,4))
        ax = fig.add_subplot(111)

        ax.plot(width_for_plot, depth_for_plot,'b-', label = 'Design')
        

        ax.set_ylim(np.max(depth_for_plot) + np.max(depth_for_plot)*0.05,-0.05)
        ax.set_xlabel('Length [mm]',fontsize = 12)
        ax.set_ylabel('Depth [mm]',fontsize = 12)
        ax.grid()
        fig.tight_layout()
        plt.show()
        plt.close()
        
    def SuperEllipse_hone(self,h_arr,rx_arr,ry_arr,nx_arr,ny_arr,p2, p_int, Plot = False):
        # ==========================
        # Create 3D model of super ellipse based SWS
        # - - - - - - - - - - - - - - -                
        
        Z = np.ones([p_int,p_int])*h_arr[0]  # set depth map (zeros)[mm]
        y,x = np.linspace(-p2,p2,p_int), np.linspace(-p2,p2,p_int) # set 1D x and y array [mm]
        Y,X = np.meshgrid(y,x)                                     # set 2D x and y pixel map [mm]
        pix_size = x[1] -x[0]                                      # pixel size [mm]
                
        #  ~~ Create 3D model of super ellipse ~~
        for ind in tqdm(range(0,len(h_arr)),desc = 'SWS contour...'):            
            where_ind = np.where(abs(X/rx_arr[ind])**(nx_arr[ind]) + abs(Y/ry_arr[ind])**(ny_arr[ind]) <= 1)
            Z[where_ind] = h_arr[ind]
            
        Z_off = -(Z - np.max(h_arr))
        
        if Plot == True:
            fig = plt.figure(figsize = (6,6))
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.contour(X*1e+3,Y*1e+3,Z_off*1e+3,cmap = 'jet_r')
            ax.tick_params(labelsize = 12)
            ax.set_xlabel('$x$ [mm]',fontsize = 15)
            ax.set_ylabel('$y$ [mm]',fontsize = 15)
            cm = ax.pcolormesh(X*1e+3,Y*1e+3,Z_off*1e+3,cmap = 'jet_r',alpha = 0.8)
            fig.colorbar(cm,shrink = 0.7,label= 'Height [mm]')
            fig.tight_layout()
                    
        return X,Y,Z_off
    
    
    def Datwrite(self,fname,m,z_res):
        '''
        * Def name
            Datwrite
        * Description
            Create 3D index array for RCWA calculation consisting of [0,1]
        * input parameters
            - fname: file name for .dat
            - m: 3D shape of SWS
        * return
            - 
        '''
        f = open('./Data/'+fname+'.dat','w') # file open
        
        # ==========================
        # Header for RCWA
        # - - - - - - - - - - - - - - - 
        f.write('/rn,a,b/nx0\n')
        f.write('/r,qa,qb\n')
        f.write('/r\n')
        f.write(str(len(m[0])) + ' -1 1 Z_DEPENDENT OUTPUT_REAL_3D\n')
        f.write(str(len(m)) + ' -1 1\n')
        f.write(str(z_res) + ' 0 1\n') 
        # - - - - - - - - - - - - - - - 
        # ==========================
        
        z_max = np.max(m)   # find maximum m
        z_min = np.min(m)   # find minimum m
        dz = (z_max-z_min)/float(z_res)        # calculate thickness of each layer

        # ==========================
        # find substrate area for each layer
        # - - - - - - - - - - - - - - - 
        for k in tqdm(range(z_res),desc = 'RCWA input...'):  
            for i in range(len(m[0])):
                for j in range(len(m)):
                    if( m[j][i] <= z_max - (float(k)+0.5)*dz ):
                        f.write('0.0 ')
                    else:
                        f.write('1.0 ')
                f.write('\n')
        f.close()
        # - - - - - - - - - - - - - - - 
        # ==========================        
        
        
class VK4_Lib:
    def getscale(self):
        with open(self.fpas, 'rb') as f:
            header = f.read(264)
        XY = int(binascii.hexlify(header[252:256][::-1]),16)/1e9
        Z = int(binascii.hexlify(header[260:264][::-1]),16)/1e9
        return XY, Z

    def heightviewer(self):
        with open(self.fpas, 'rb') as f:
            header = f.read(40)
        with open(self.fpas, 'rb') as f:
            offset = int(binascii.hexlify(header[36:40][::-1]),16)
            f.seek(offset)
            pre_data = f.read(28)
            width = int(binascii.hexlify(pre_data[0:4][::-1]),16)
            height = int(binascii.hexlify(pre_data[4:8][::-1]),16)
            databytes = int(int(binascii.hexlify(pre_data[8:12][::-1]),16)/8)
            totlength = int(binascii.hexlify(pre_data[16:20][::-1]),16)
            LZWtable = f.read(256*3)
            data = f.read(width*height*databytes)
        data_array = np.empty(width*height)
        for i in range(width*height):
            data_array[i] = int(binascii.hexlify(data[i*databytes:(i+1)*databytes][::-1]),16)
        image = np.reshape(data_array,(height,width))
        return image    
    
class Shape_evaluation(VK4_Lib):
    def __init__(self,dir,fname,offx,offy,tx,ty,ex,ey):
        self.fs = 15
        self.pas = './'
        self.date = dir
        self.fname = fname
        self.fpas = self.pas+self.date+'/'+self.fname+'.vk4'
        self.xy_cal = self.getscale()[0]
        self.Z_cal = self.getscale()[1]
        print('Z_cal = ',self.Z_cal)
        print('xy_cal = ',self.xy_cal)
        self.m = self.heightviewer()
        self.m = (self.m - np.max(self.m))*1e-7
        
        self.m = self.m[tx:ex,ty:ey]
        self.x_com = len(self.m[0])
        self.y_com = len(self.m)

        self.x = np.linspace(0,self.x_com*self.xy_cal,self.x_com)
        self.y = np.linspace(0,self.y_com*self.xy_cal,self.y_com)
    
        self.X,self.Y = np.meshgrid(self.x,self.y)
        
        self.theta = np.linspace(0,np.pi*2,1001)

    
    def Find_one_structure(self,p,cuty,cutx):
        """
        Find one structures by finding minimum location of the groove
        Input parameters:
            p: pitch of the structures [mm]
            cuty: initial cut in y (int)
            cutx: initial cut in x (int)
        Return m_cut: one structure (2-D numpy array)
        """
        p_int = int(p/self.xy_cal) # calculate index number for the actual pitch
        
        m_cuty = self.m[cuty:cuty+p_int]   # m cut in y direction 
        m_cutx = self.m[:,cutx:cutx+p_int] # m cut in x direction
        
        locy = int(np.mean(np.argmin(m_cuty,axis = 0))) # avarage minimum location in y
        locx = int(np.mean(np.argmin(m_cutx,axis = 1))) # average minimum location in x
        
        m_cut = self.m[locy:locy+p_int,locx:locx+p_int] # m one structure
        return m_cut
    
    def Find_contour(self,m_cut,z_pos,p):
        m_cont = np.where(m_cut>=z_pos)
        # number of pixel for one structure pitch
        p_int = int(p/self.xy_cal)
        
        img = np.zeros((len(m_cut),len(m_cut[0])))
        img = img.astype(np.uint8)

        img[m_cont[0],m_cont[1]]=255

        contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        cont_arr = np.empty(len(contours),dtype = int)
        for i in range(len(contours)):
            cont_arr[i] = len(contours[i])
        #print('Contours array =', cont_arr)
        cont_num = np.argmax(cont_arr)
        #print('SWS contour number = ',cont_num)
        
        arr = np.array(contours[cont_num])[:,0]

        x_box = np.linspace(-p/2,p/2,p_int)
        y_box = np.linspace(-p/2,p/2,p_int)
        x_data = x_box[arr[:,0]]
        y_data = y_box[arr[:,1]]
        
        return x_data,y_data

    def Fit_contours(self,x_data,y_data,p):
        def func(data,rx,ry,nx,ny,offx,offy):
            x,y = data
            return (np.absolute(x-offx)/rx)**nx + (np.absolute(y-offy)/ry)**ny

        rx_init = (np.max(x_data)-np.min(x_data))/2
        ry_init = (np.max(y_data)-np.min(y_data))/2
        nx_init = 3
        ny_init = 3
        offx_init = (np.max(x_data)+np.min(x_data))/2
        offy_init = (np.max(y_data)+np.min(y_data))/2

        #print(rx_init,ry_init,offx_init,offy_init)
        bounds_arr = ((rx_init*0.001,ry_init*0.001,2.0,2.0,-p/10,-p/10),(p/2,p/2,500,500,p/10,p/10))
        
        popt,pcov = curve_fit(func,(x_data,y_data),
                              np.ones(len(x_data)),
                              p0=[rx_init,ry_init,nx_init,ny_init,offx_init,offy_init],
                              bounds=bounds_arr)
        
        lib1 = Gen_SuperEllipse()
        x,y = lib1.Contour_SuperEllipse(popt[0],popt[1],popt[2],popt[3])
        
        return popt,x+popt[4], y+popt[5]
    
    def Output_fitted_parameters(self,m_cut,z_pos_arr,p, nonoff,sname):

        rx_fit = np.empty(len(z_pos_arr)-1)
        ry_fit = np.empty(len(z_pos_arr)-1)
        nx_fit = np.empty(len(z_pos_arr)-1)
        ny_fit = np.empty(len(z_pos_arr)-1)
        offx_fit = np.empty(len(z_pos_arr)-1)
        offy_fit = np.empty(len(z_pos_arr)-1)

        x_data_arr = []
        y_data_arr = []

        x_fit_arr = np.empty((len(z_pos_arr)-1,1001))
        y_fit_arr = np.empty((len(z_pos_arr)-1,1001))

        for zi in range(0,len(z_pos_arr)):
            x_data,y_data = self.Find_contour(m_cut,z_pos_arr[zi],p)
            popt,x_fit,y_fit = self.Fit_contours(x_data,y_data,p)

            x_data_arr.append(x_data)
            y_data_arr.append(y_data)
            rx_fit[zi-1] = popt[0]
            ry_fit[zi-1] = popt[1]
            nx_fit[zi-1] = popt[2]
            ny_fit[zi-1] = popt[3]
            offx_fit[zi-1] = popt[4]
            offy_fit[zi-1] = popt[5]

            x_fit_arr[zi-1] = x_fit
            y_fit_arr[zi-1] = y_fit


        fig = plt.figure(figsize = (12,8))
        gs = GridSpec(3,4)

        ss1 = gs.new_subplotspec((0,0),rowspan = 2,colspan = 2)  # contour
        ss2 = gs.new_subplotspec((0,2),colspan = 2)              # radius (taper)
        ss3 = gs.new_subplotspec((1,2),colspan = 2)              # index (taper)
        ss4 = gs.new_subplotspec((2,2),colspan = 2)              # index (taper)

        ax1 = plt.subplot(ss1) # contour
        ax2 = plt.subplot(ss2) # radius (taper)
        ax3 = plt.subplot(ss3) # index (taper)# color gradation
        ax4 = plt.subplot(ss4) # index (taper)# color gradation


        for i in range(1,len(x_data_arr),13):
            ax1.plot(x_data_arr[i],y_data_arr[i],'.',color = cm.jet_r(i/len(x_data_arr)), markersize = 2)
        for j in range(1,len(x_fit_arr),13):
            ax1.plot(x_fit_arr[j],y_fit_arr[j],color = cm.jet_r(j/len(x_fit_arr)),alpha = 0.7)

        ax1.grid(True)
        ax1.set_xlabel('x [mm]',fontsize = 14)
        ax1.set_ylabel('y [mm]', fontsize = 14)
        ax1.set_aspect(True)
        ax1.tick_params(labelsize = 12)

        ax2.plot(z_pos_arr[1+nonoff:]*(-1),rx_fit[nonoff:],'r-',label = r'$r_x$')
        ax2.plot(z_pos_arr[1+nonoff:]*(-1),ry_fit[nonoff:],'b-',label = r'$r_y$')
        ax2.set_xlabel('Depth [mm]',fontsize = 14)
        ax2.set_ylabel('Radians [mm]',fontsize = 14)
        ax2.set_ylim(0.0,p/2*1.1)
        ax2.tick_params(labelsize = 12)
        ax2.grid(True)
        ax2.legend()


        ax3.set_yscale('log')
        ax3.plot(z_pos_arr[1+nonoff:]*(-1),nx_fit[nonoff:],'r-',label = r'$n_x$')
        ax3.plot(z_pos_arr[1+nonoff:]*(-1),ny_fit[nonoff:],'b-',label = r'$n_y$')
        ax3.set_xlabel('Depth [mm]',fontsize = 14)
        ax3.set_ylabel('Index [mm]',fontsize = 14)
        ax3.tick_params(labelsize = 12)
        ax3.grid(True)
        ax3.legend()


        ax4.plot(z_pos_arr[1+nonoff:]*(-1),offx_fit[nonoff:],'r-',label = r'$offset_x$')
        ax4.plot(z_pos_arr[1+nonoff:]*(-1),offy_fit[nonoff:],'b-',label = r'$offset_y$')
        ax4.set_xlabel('Depth [mm]',fontsize = 14)
        ax4.set_ylabel('Offset [mm]',fontsize = 14)
        ax4.tick_params(labelsize = 12)
        ax4.grid(True)
        ax4.legend()

        fig.tight_layout()
        plt.savefig('./Figure/'+sname+'_fit_sum.png',dpi = 300)
        plt.show()
        plt.close()
        
        
        X,Y,Z = self.SuperEllipse_hone(z_pos_arr[:-(1+nonoff)],
                              rx_fit[::-1][nonoff:],
                              ry_fit[::-1][nonoff:],
                              nx_fit[::-1][nonoff:],
                              ny_fit[::-1][nonoff:],
                              offx_fit[::-1][nonoff:],
                              offy_fit[::-1][nonoff:],p/2)
        return X,Y,Z
        
    
    def SuperEllipse_hone(self,h_arr,rx_arr,ry_arr,nx_arr,ny_arr,offx,offy,p2):
        # ==========================
        # Create 3D model of super ellipse based SWS
        # - - - - - - - - - - - - - - -        
        p_int = int(p2*2/self.xy_cal)
        
        
        Z = np.ones([p_int,p_int])*h_arr[0]  # set depth map (zeros)[mm]
        y,x = np.linspace(-p2,p2,p_int), np.linspace(-p2,p2,p_int) # set 1D x and y array [mm]
        Y,X = np.meshgrid(y,x)                                     # set 2D x and y pixel map [mm]
        pix_size = x[1] -x[0]                                      # pixel size [mm]
                
        #  ~~ Create 3D model of super ellipse ~~
        for ind in tqdm(range(0,len(h_arr)),desc = 'SWS contour...'):            
            where_ind = np.where(abs((X - offx[ind])/rx_arr[ind])**(nx_arr[ind]) + abs((Y - offy[ind])/ry_arr[ind])**(ny_arr[ind]) <= 1)
            Z[where_ind] = h_arr[ind]
                    
        return X,Y,Z  
    
    
    def Datwrite(self,fname,m,z_res):
        '''
        * Def name
            Datwrite
        * Description
            Create 3D index array for RCWA calculation consisting of [0,1]
        * input parameters
            - fname: file name for .dat
            - m: 3D shape of SWS
        * return
            - 
        '''
        f = open('./Data/'+fname+'.dat','w') # file open
        
        # ==========================
        # Header for RCWA
        # - - - - - - - - - - - - - - - 
        f.write('/rn,a,b/nx0\n')
        f.write('/r,qa,qb\n')
        f.write('/r\n')
        f.write(str(len(m[0])) + ' -1 1 Z_DEPENDENT OUTPUT_REAL_3D\n')
        f.write(str(len(m)) + ' -1 1\n')
        f.write(str(z_res) + ' 0 1\n') 
        # - - - - - - - - - - - - - - - 
        # ==========================
        
        z_max = np.max(m)   # find maximum m
        z_min = np.min(m)   # find minimum m
        dz = (z_max-z_min)/float(z_res)        # calculate thickness of each layer

        # ==========================
        # find substrate area for each layer
        # - - - - - - - - - - - - - - - 
        for k in tqdm(range(z_res),desc = 'RCWA input...'):  
            for i in range(len(m[0])):
                for j in range(len(m)):
                    if( m[j][i] <= z_max - (float(k)+0.5)*dz ):
                        f.write('0.0 ')
                    else:
                        f.write('1.0 ')
                f.write('\n')
        f.close()
        # - - - - - - - - - - - - - - - 
        # ==========================
        


class Gen_SuperEllipse_SWS:
    '''
    General class with respect to modeling super-ellipse based SWS
    Super-ellipse based SWS can be similar shape with laser-ablated SWS, then this design can be reflected to evaluation of actual fabrication
    
    Here is a list of fuctions in the class:
    
    * Def name
        __init__
    * Description
        set global values
    * input parameters
        - theta_res: resolution of contour of super ellipse as int.
        - xy_ind: resolution of pixeling of 3D shape of SWS as int.
        - z_res: resolution of depth of SWS as int.
        - input_freq: input frequency [Hz]
        - input_freq_band: input frequency band [Hz]

    * Defined global values
        - data_path: path to data directory  (need to modify with your location)
        - fig_path: path to figure directory (need to modify with your location)
        - mu:  vacuum permeability [m kg s-2 A-2]
        - ep0: vacuum permitivity [m-3 kg-1 s4 A2]
        - c:   speed of light [m/s]
        - pi:  pi
        ~~ for only 90 - 150 GHz ~~
        - freq_Hz: input frequency [Hz]
        - vc:      center frequency [Hz]
        - fb:      fractional bandwidth
        - vb: lower edge of frequency bands [Hz]
        - vu: upper edge of frequency bands [Hz]
        - band90_index: index array where values of other array are included at 90 GHz frequency band
        - band150_index: index array where values of other array are included at 150 GHz frequency band
    
    '''
    def __init__(self,theta_res,xy_ind,z_res,input_freq,input_freq_band):        
        # ==========================
        # Resolution and file pathes
        # - - - - - - - - - - - - - - -
        self.theta = np.linspace(0,np.pi*2,theta_res)  # resolution of contour of super ellipse
        self.xy_ind = xy_ind                           # resolution of pixeling of 3D shape of SWS
        self.z_res = z_res                             # resolution of depth of SWS
        self.data_path = '../Complex_SWS/Data/'        # path to data directory  (need to modify with your location)
        self.fig_path = '../Complex_SWS/Figure/'       # path to figure directory (need to modify with your location)
        # - - - - - - - - - - - - - - -
        # ==========================
        
        # ==========================
        # Constants
        # - - - - - - - - - - - - - - -
        self.mu = 1.25663706e-06  # vacuum permeability [m kg s-2 A-2]
        self.ep0 = 8.8542e-12     # vacuum permitivity [m-3 kg-1 s4 A2]
        self.c = 2.9979e+08       # speed of light [m/s]
        self.pi = np.pi           # pi
        # - - - - - - - - - - - - - - -
        # ==========================
        
        # ==========================
        # For 90-150 Filter
        # - - - - - - - - - - - - - - -
        self.freq_Hz = input_freq    # input frequency [Hz]
        self.vc = input_freq_band        # center frequency [Hz]            
        self.fb = 0.3                              # fractional bandwidth
        self.vb = self.vc - (self.vc*self.fb/2.)   # lower edge of frequency bands [Hz]
        self.vu = self.vc + (self.vc*self.fb/2.)   # upper edge of frequency bands [Hz]
        # find index array of the bands
        self.band90_index = np.where( (self.freq_Hz > self.vb[0]) & (self.freq_Hz < self.vu[0]) ) 
        self.band150_index = np.where( (self.freq_Hz > self.vb[1]) & (self.freq_Hz < self.vu[1]) ) 
        # - - - - - - - - - - - - - - -
        # ==========================
        
    def Klopfenstein(self,h,num,ni,ns,Gamma):
        '''
        * Def name
            Klopfenstein
        * Description
            Calculate Klofenstein index profile
            Based on two papers:
                1: Klopfenstein(1956):https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4051841
                2: Grann et al(1995): https://opg.optica.org/view_article.cfm?pdfKey=fbf947c3-8dd3-440b-bfe51c58d6e98f89_33114
        * input parameters
            - h:   height of the taper along z axis [arbitral unit]
            - num: number of layer (as same as resolution of depth of SWS)
            - ni:  refractive index of air, so 1.0
            - ns:  refractive index of substrate
            - Gamma: important parameter to control the trade-off between the ripple in the opera- tion bands and bandwidth

        * return
            - n_arr: Klopfenstein index profile
            - z_arr: array of depth (height)
            - z_space: array of thickness each layer
        '''
        z_arr = np.linspace(0,h,num)            # define array of depth (height)
        n_arr = np.zeros(num)                   # set array of Klopfenstein index profile
        z_space = np.ones(len(z_arr))*(h/num)   # calculate array of thickness each layer
        rho_0 = 0.5*np.log(ns/ni)               # calcualte rho_0(see Klopfenstein(1956))
        A = np.arccosh(rho_0/Gamma)             # calculate A(see Klopfenstein(1956))
        
        # Calculate Klopfenstein index profile 
        # (see Eq.12 in Klopfenstein(1956), and Eq.6 in Grann et al(1995))
        for i in range(0,num):
            x_i = 2.*z_arr[i]/h-1.
            phi_int = integrate.quad(lambda y: special.i1(A*np.sqrt(1.-y**2))/(A*np.sqrt(1.-y**2)), 0, x_i)  
            n_arr[i] = np.sqrt(ni*ns) * np.exp(Gamma * A**2 * phi_int[0])
        return n_arr,z_arr,z_space
        
    def Brauer_emt_anti_symmetric(self,freq,n1,n2,f1,f2,p1,p2):
        '''
        * Def name
            Brauer_emt_anti_symmetric
        * Description
            Calculate effective refractive index given area fraction
            Based on Brauer(1994): https://opg.optica.org/view_article.cfm?pdfKey=92654043-e0c1-4520-98b312f00908c33d_42237
        * input parameters
            - freq:   inpu frequency [Hz]
            - n1: refractive index of air, so 1.0
            - n2: refractive index of substrate
            - f1: area fraction along x axis
            - f2: area fraction along y axis
            - p1: pitch x
            - p2: pitch y

        * return
            - n_: 0th ordered effective refractive index
            - neff: 2nd ordered effective refractive index
        '''
        
        lamda = self.c/freq   # wavelength
        f = (f1+f2)/2.        # average fraction in x and y

        e1 = n1**2.*self.ep0  # refractive index --> permitivity (air) 
        e2 = n2**2.*self.ep0  # refractive index --> permitivity (substrate)

        ell_0 = (1.0 - f1)*e1+f1*e2    # Eq.1
        els_0 = 1./((1.-f2)/e1+f2/e2)  # Eq.2

        ell_2 = ell_0*(1.+(np.pi**2/3.)*(p1/lamda)**2.*f1**2*(1.-f1)**2.*((e2-e1)**2./(self.ep0*ell_0)))                    # Eq.3
        els_2 = els_0*(1.0+(np.pi)**2/3.0*(p2/lamda)**2*f2**2*(1.-f2)**2.*((e2-e1)**2.)*ell_0/self.ep0*(els_0/(e2*e1))**2.) # Eq.4
        
        e_2nd_up = (1.0 - f1)*e1 + f1*els_2           # Eq.6
        e_2nd_down = 1./((1.0 - f2)/e1 + f2/ell_2)    # Eq.7

        n_=(1-f**2)*n1+f**2*n2                        # Eq.5
        n__2nd_up = np.sqrt(e_2nd_up/(self.ep0))      # permitivity --> refractive index (up)
        n__2nd_down = np.sqrt(e_2nd_down/(self.ep0))  # permitivity --> refractive index (down)

        neff = 0.2*(n_+2.0*n__2nd_up+2.0*n__2nd_down) # Eq.8
        return n_*np.ones(len(neff)),neff
    
    
    
    def fit_oblique_basic_multilayer_r_t_incloss(self, n, losstan, d, freq_in, angle_i, incpol):
        '''
        * Def name
            fit_oblique_basic_multilayer_r_t_incloss
        * Description
            Calculate coefficient of reflectance and transmittance based on Transfer matrix method, made by Tomo Matsumura
            (modified with this class by RTakaku)
        * input parameters
            - n:  refractive index of substrate
            - losstan: loss tangent of substrate
            - d: thickness [m] 
            - freq_in: input frequency [Hz]
            - angle_i: incident angle [rad]
            - incpol: 1 for s-state, E field perpendicular to the plane of incidnet, -1 for P-state, E in the plane of incident

        * return
            - output ([0]: freq, [1]: coeff of reflectance, [2]: coeff of transmittance (complex numpy array))
        '''
        num=len(d) #; the number of layer not including two ends
        const = np.sqrt((8.85e-12)/(4.*self.pi*1e-7)) #SI unit sqrt(dielectric const/permiability)

        # ;-----------------------------------------------------------------------------------
        # ; angle of refraction
        angle = np.zeros(num+2)          # ; angle[0]=incident angle
        angle[0] = angle_i
        for i in range(0,num+1): angle[i+1] = np.arcsin(np.sin(angle[i])*n[i]/n[i+1])

        # ;-----------------------------------------------------------------------------------
        # ; define the frequency span
        l = len(freq_in)
        output = np.zeros((3,l),'complex') # output = dcomplexarr(3,l)

        # ;-----------------------------------------------------------------------------------
        # ; define the effective thickness of each layer
        h = np.zeros(num,'complex')
        n_comparr = np.zeros(len(n),'complex')
        n_comparr[0] = complex(n[0], -0.5*n[0]*losstan[0])
        n_comparr[num+1] = complex(n[num+1], -0.5*n[num+1]*losstan[num+1])

        # ;-----------------------------------------------------------------------------------
        # ; for loop for various thickness of air gap between each layer
        for j in range(0,l):
            for i in range(0,num): 
                n_comparr[i+1] = complex(n[i+1], -0.5*n[i+1]*losstan[i+1])
                h[i] = n_comparr[i+1]*d[i]*np.cos(angle[i+1]) # ;effective thickness of 1st layer

            freq = freq_in[j]
            k = 2.*self.pi*freq/self.c

            # ;===========================================
            # ; Y: Y[0]=vacuum, Y[1]=1st layer..., Y[num+1]=end side
            Y = np.zeros(num+2,'complex')
            for i in range(0,num+2):
                if (incpol == 1):
                    Y[i] = const*n_comparr[i]*np.cos(angle[i])
                    cc = 1.
                if (incpol == -1):
                    Y[i] = const*n_comparr[i]/np.cos(angle[i])
                    cc = np.cos(angle[num+1])/np.cos(angle[0])

            # ;===========================================
            # ; define matrix for single layer
            m = np.identity((2),'complex')    # ; net matrix
            me = np.zeros((2,2),'complex') # ; me[0]=1st layer, ...
            for i in range(0,num):
                me[0,0] = complex(np.cos(k*h[i]), 0.)
                me[1,0] = complex(0., np.sin(k*h[i])/Y[i+1])
                me[0,1] = complex(0., np.sin(k*h[i])*Y[i+1])
                me[1,1] = complex(np.cos(k*h[i]), 0.)
                m = np.dot(m,me)

            r = (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]-m[0,1]*cc-Y[num+1]*m[1,1]) / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])
            t = 2.*Y[0] / (Y[0]*m[0,0]*cc+Y[0]*Y[num+1]*m[1,0]+m[0,1]*cc+Y[num+1]*m[1,1])

            output[0,j] = freq+0.j #; unit of [Hz]
            output[1,j] = r
            output[2,j] = t

        return output

    def b_factor(self,a,p2):
        '''
        * Def name
            b_factor
        * Description
            One condition for radius in bottom ellipse y to keep the same total depth even with setting asymmetric SWS in x and y
        * input parameters
            - a: radius in bottom ellipse x
            - p2: pitch/2 
        * return
            - b (radius in bottom ellipse y)
        '''
        return a*p2/(np.sqrt(a**2 - p2**2)) 
    
    def SuperEllipse_hone(self,input_n,input_losstan,input_d,Gamma,h,r_x_top,r_y_top,r_x_bottom,r_y_bottom,alpha_x,alpha_y,nx,ny,beta_x,beta_y,p2,sgeo_name):
        '''
        * Def name
            SuperEllipse_hone
        * Description
            - Main function
            - Make 3D model, cross section, radius(z), index(z) of super ellipse 
            - Calculate area fraction, effective index profile, and transmission of super ellipse 
            - Compare with Klopfenstein index profile and its transmission
        * input parameters
            - input_n: refractive index of substrate
            - input_losstan: loss tangent of substrate
            - input_d: thickness of substrate (not included SWS depth)
            - Gamma: Gamma used in Grann et al. (1994)
            - h: total depth (height) of structure [mm]
            - r_x_top: top radius in x [mm]
            - r_y_top: top radius in y [mm]
            - r_x_bottom: bottom radius in x [mm]
            - r_y_bottom: bottom radius in y [mm]
            - alpha_x: curvature of SWS in x
            - alpha_y: curvature of SWS in y
            - nx: index of super ellipse in x
            - ny: index of super ellipse in y
            - beta_x: one parameter to change nx profile in x
            - beta_y: one parameter to change ny profile in y
            - p2: pitch/2 [mm]
            - sgeo_name: file name of .npz file which includes all of created data in this function
        * return
            - X, Y, Z: 3D map of one super ellise based SWS
        '''
        
        # ==========================
        # Calculate Klopfenstein index profile and its transmission
        # - - - - - - - - - - - - - - -
        n_arr, z_klop, d_klop = lib.Klopfenstein(h*1e-3,100,1.0,input_n,Gamma) # n, z, thickness each layer
        n_klop = np.concatenate((np.array([1.0]),n_arr))                       # just add air layer (incident environment) to n_arr, which is exactly Klopfenstein index profile

        thickness_for_klop = np.concatenate((d_klop,np.array([input_d]),d_klop))   # thickness array ( Klopfesntein array (N of layer) + substrate + Klopfesntein array (N of layer) )   
        index_for_klop = np.concatenate((n_klop,np.array([input_n]),n_klop[::-1])) # refractive index array ( air + Klopfesntein array (N of layer) + substrate + Klopfesntein array (N of layer) + air)
        losstan_for_klop = np.concatenate((np.ones(len(n_klop))*input_losstan,np.array([input_losstan]),np.ones(len(n_klop))*input_losstan)) 
                                                                                   # loss tangent array ( air + Klopfesntein array (N of layer) + substrate + Klopfesntein array (N of layer) + air)

        trans_klop = abs(lib.fit_oblique_basic_multilayer_r_t_incloss(index_for_klop, losstan_for_klop, thickness_for_klop, self.freq_Hz,angle_i=0, incpol=1)[2])**2 
                                                                                   # This indicates |trans. coeff|**2
        
        # Now, we have n_klop(Klopfenstein index profile), and trans_klop (Transmission based on Klofenstein ARC)
        # - - - - - - - - - - - - - - -
        # ==========================
        
        # ==========================
        # Create 3D model of super ellipse based SWS
        # - - - - - - - - - - - - - - -        
        Z = np.zeros([self.xy_ind,self.xy_ind])                                  # set depth map (zeros)[mm]
        x,y = np.linspace(-p2,p2,self.xy_ind), np.linspace(-p2,p2,self.xy_ind)   # set 1D x and y array [mm]
        X,Y = np.meshgrid(x,y)                                                   # set 2D x and y pixel map [mm]
        pix_size = x[1] -x[0]                                                    # pixel size [mm]
        
        h_arr = np.linspace(0,h,self.z_res)                                      # depth (height) array [mm]
        
        r_x_taper = r_x_top + (r_x_bottom-r_x_top)*(1-(h_arr/h)**alpha_x)        # calculate radius taper x along z [mm]
        r_y_taper = r_y_top + (r_y_bottom-r_y_top)*(1-(h_arr/h)**alpha_y)        # calculate radius taper y along z [mm]
        
        fac = np.linspace(0,1.0,self.z_res)          # factor for index of super ellipse
        nx_arr = (nx[1]-nx[0])*fac**beta_x + nx[0]   # calculate index taper of super ellipse x along z
        ny_arr = (ny[1]-ny[0])*fac**beta_y + ny[0]   # calculate index taper of super ellipse y along z
        
        Area_fraction = np.zeros(self.z_res)         # area fraction (zeros)
        
        #  ~~ Create 3D model of super ellipse ~~
        for ind in tqdm(range(0,len(h_arr)),desc = 'SWS contour...'):
            r_x_i = r_x_taper[ind]  # rx (z)
            r_y_i = r_y_taper[ind]  # ry (z)

            xi = r_x_i * np.sign(np.cos(self.theta))*abs(np.cos(self.theta))**(2/nx_arr[ind]) # rx (array to make contour)
            yi = r_y_i * np.sign(np.sin(self.theta))*abs(np.sin(self.theta))**(2/ny_arr[ind]) # ry (array to make contour)
            
            for i in range(0,len(xi)):
                where_ind = np.where((abs(X) <= abs(xi[i])) & (abs(Y) <= abs(yi[i])))  # pick up area (N of pixel here) within contour of SWS at given z
                Z[where_ind] = h_arr[ind]                                              # define depth (height) for area within contour of SWS at given z
            count_area = np.count_nonzero(Z >= h_arr[ind])                             # count N of pixel within contour of SWS at given z
            Area_fraction[ind] = (count_area*pix_size**2)/(len(x)*len(x)*pix_size**2)  # calculate area fraction of one unit structure at given z, which is used to calculate effective refractive index
        
        #  ~~ Calculate effective refractive index at 90 GHz ~~            
        n_emt = self.Brauer_emt_anti_symmetric(90e+9,1.0,input_n,Area_fraction,Area_fraction,p*1e-3,p*1e-3) # calculate effective refractive index of super ellipse based SWS using Area_fraction, only at 90 GHz just to show the example of the taper
        
        # ~~ Calculate transmittance ~~
        trans_arr = np.zeros(len(self.freq_Hz))  # transmittance (zero) array
        for freq_i in tqdm(range(0,len(self.freq_Hz)),desc ='transmission...'):
            n_emt_i = self.Brauer_emt_anti_symmetric(self.freq_Hz[freq_i],1.0,input_n,Area_fraction,Area_fraction,p*1e-3,p*1e-3)[1]   # calculate effective refractive index of super ellipse based SWS at given frequency [Hz]
            eff_n_arr = np.concatenate((np.array([1.0]),n_emt_i[::-1],np.array([input_n]),n_emt_i,np.array([1.0])))                   # refractive index array (air + n_emt + substrate + n_emt + air)
            losstan_arr = np.concatenate((np.array([0.]),np.ones(len(n_emt_i))*input_losstan,np.array([input_losstan]),np.ones(len(n_emt_i))*input_losstan,np.array([0.]))) # loss tangent array (air + n_emt + substrate + n_emt + air)
            d_arr = np.concatenate((np.ones(len(n_emt_i))*(h_arr[1]-h_arr[0])*1e-3,np.array([input_d]),np.ones(len(n_emt_i))*(h_arr[1]-h_arr[0])*1e-3))                     # thickness array (n_emt + substrate + n_emt)
            trans_output = self.fit_oblique_basic_multilayer_r_t_incloss(eff_n_arr, losstan_arr, d_arr, np.array([self.freq_Hz[freq_i]]), angle_i=0, incpol=1)[2]           # trans. coeff
            trans_arr[freq_i] = abs(trans_output)**2
                
        # ~~ save all data as .npz ~~
        np.savez(self.data_path+sgeo_name+'.npz',
                 xy_res = self.xy_ind, z_res = self.z_res,
                 p = p, h = h, h_arr = h_arr,
                 r_x_top = r_x_top, r_y_top = r_y_top, 
                 r_x_bottom = r_x_bottom, r_y_bottom = r_y_bottom, 
                 r_x_taper = r_x_taper, r_y_taper = r_y_taper,
                 alpha_x = alpha_x, alpha_y = alpha_y, 
                 nx = nx, ny = ny, nx_arr = nx_arr, ny_arr = ny_arr,
                 beta_x = beta_x, beta_y = beta_y,
                 x = x,y = y,
                 X = X, Y = Y, Z = Z, 
                 afrac = Area_fraction, n_emt = n_emt, n_klop = n_klop, z_klop = z_klop*1e+3,
                 freq_GHz = self.freq_Hz*1e-9,trans = trans_arr, trans_klop = trans_klop)
        
        return X,Y,Z
    
    
    def SuperEllipse_sym(self,h,r_taper,n_arr,beta,p2,sgeo_name):
        '''
        * Def name
            SuperEllipse_sym
        * Description
            - Make 3D model, cross section, radius(z), index(z) of super ellipse 
            - Calculate area fraction, effective index profile, and transmission of super ellipse
        * input parameters
            - h: total depth (height) of structure [mm]
            - r_taper: radius taper along z axis [mm]
            - n_arr: index of super ellipse
            - beta: one parameter to change n profile
            - p2: pitch/2 [mm]
            - sgeo_name: file name of .npz file which includes all of created data in this function
        * return
            - X, Y, Z: 3D map of one super ellise based SWS
        '''
                                                                                   # This indicates |trans. coeff|**2
        
        # Now, we have n_klop(Klopfenstein index profile), and trans_klop (Transmission based on Klofenstein ARC)
        # - - - - - - - - - - - - - - -
        # ==========================
        
        # ==========================
        # Create 3D model of super ellipse based SWS
        # - - - - - - - - - - - - - - -        
        Z = np.zeros([self.xy_ind,self.xy_ind])                                  # set depth map (zeros)[mm]
        x,y = np.linspace(-p2,p2,self.xy_ind), np.linspace(-p2,p2,self.xy_ind)   # set 1D x and y array [mm]
        X,Y = np.meshgrid(x,y)                                                   # set 2D x and y pixel map [mm]
        pix_size = x[1] -x[0]                                                    # pixel size [mm]
        h_arr = np.linspace(0,h,self.z_res)                                      # depth (height) array [mm]
        r_x_taper = r_taper
        r_y_taper = r_taper

        fac = np.linspace(0,1.0,self.z_res)          # factor for index of super ellipse
        nx_arr = n_arr
        ny_arr = n_arr                         # calculate index taper of super ellipse y along z
        
        Area_fraction = np.zeros(self.z_res)         # area fraction (zeros)
        
        #  ~~ Create 3D model of super ellipse ~~
        for ind in tqdm(range(0,len(h_arr)),desc = 'SWS contour...'):
            r_x_i = r_x_taper[ind]  # rx (z)
            r_y_i = r_y_taper[ind]  # ry (z)

            xi = r_x_i * np.sign(np.cos(self.theta))*abs(np.cos(self.theta))**(2/nx_arr[ind]) # rx (array to make contour)
            yi = xi
            
            for i in range(0,len(xi)):
                where_ind = np.where((abs(X) <= abs(xi[i])) & (abs(Y) <= abs(yi[i])))  # pick up area (N of pixel here) within contour of SWS at given z
                Z[where_ind] = h_arr[ind]                                              # define depth (height) for area within contour of SWS at given z
            count_area = np.count_nonzero(Z >= h_arr[ind])                             # count N of pixel within contour of SWS at given z
            Area_fraction[ind] = (count_area*pix_size**2)/(len(x)*len(x)*pix_size**2)  # calculate area fraction of one unit structure at given z, which is used to calculate effective refractive index
                
        # ~~ save all data as .npz ~~
        np.savez(self.data_path+sgeo_name+'_sym_top.npz',
                 h = h, h_arr = h_arr,
                 r_x_taper = r_x_taper, r_y_taper = r_y_taper,
                 nx_arr = nx_arr, ny_arr = ny_arr,
                 beta = beta,
                 x = x,y = y,
                 X = X, Y = Y, Z = Z, 
                 afrac = Area_fraction)        
        return X,Y,Z
    
    def Datwrite(self,fname,m):
        '''
        * Def name
            Datwrite
        * Description
            Create 3D index array for RCWA calculation consisting of [0,1]
        * input parameters
            - fname: file name for .dat
            - m: 3D shape of SWS
        * return
            - 
        '''
        

        f = open(self.data_path+fname+'.dat','w') # file open
        
        # ==========================
        # Header for RCWA
        # - - - - - - - - - - - - - - - 
        f.write('/rn,a,b/nx0\n')
        f.write('/r,qa,qb\n')
        f.write('/r\n')
        f.write(str(len(m[0])) + ' -1 1 Z_DEPENDENT OUTPUT_REAL_3D\n')
        f.write(str(len(m)) + ' -1 1\n')
        f.write(str(self.z_res) + ' 0 1\n') 
        # - - - - - - - - - - - - - - - 
        # ==========================
        
        z_max = np.max(m)   # find maximum m
        z_min = np.min(m)   # find minimum m
        dz = (z_max-z_min)/float(self.z_res)        # calculate thickness of each layer

        # ==========================
        # find substrate area for each layer
        # - - - - - - - - - - - - - - - 
        for k in tqdm(range(0,self.z_res),desc = 'RCWA input...'):  
            for i in range(len(m[0])):
                for j in range(len(m)):
                    if( m[j][i] <= z_max - (float(k)+0.5)*dz ):
                        f.write('0.0 ')
                    else:
                        f.write('1.0 ')
                f.write('\n')
        f.close()
        # - - - - - - - - - - - - - - - 
        # ==========================
        
        
    def Plot_geometry(self,p,f_load,f_save):
        '''
        * Def name
            Plot_geometry
        * Description
            Plot coutour, 3D image, cross section, radius, index, area fraction, effective refractive index profile, and transmission of SWS
        * input parameters
            - p: pitch [mm]
            - f_load: name of load file
            - f_save: name of save file .png
        * return
            - X,Y,z_offset (3D data)
        '''
        
        f = np.load(lib.data_path + f_load+'.npz') # load file
        
        x = f['x']                         # length x
        X, Y, Z = f['X'], f['Y'], f['Z']   # 3D data
        z_offset = -1*(Z - np.max(Z))      # to flip z direction from height to depth


        csx = z_offset[int(len(z_offset)/2)]          # cross section view in x
        csy = z_offset[:,int(len(z_offset[:,0])/2)]   # cross section view in y

        nx, ny = f['nx'], f['ny']                     # index of super ellipse (x,y)
        beta_x, beta_y = f['beta_x'], f['beta_y']     # beta (x,y)

        r_x_top, r_y_top, r_x_bottom, r_y_bottom, f['r_x_top'], f['r_y_top'], f['r_x_bottom'], f['r_y_bottom']  # radius (top,bottom in x and y)
        alpha_x, alpha_y = f['alpha_x'], f['alpha_y']                                                           # alpha (x,y)
        
        nx_arr, ny_arr = f['nx_arr'], f['ny_arr']              # index array of super ellipse (x,y)
        r_x_taper, r_y_taper = f['r_x_taper'], f['r_y_taper']  # radius array of super ellipse (x,y)
        h_arr = f['h_arr']                                     # height (depth) [mm]
        
        afrac = f['afrac']        # Area fraction
        n_emt = f['n_emt']        # effective refractive index
        z_klop = f['z_klop']      # z Klopfenstein
        n_klop = f['n_klop']      # Klopfenstein index profile
        
        freq = f['freq_GHz']         # input frequency [GHz]
        trans = f['trans']           # transmission from EMT 
        trans_klop = f['trans_klop'] # transmission of Klopfenstein index profile
        
        
        clevel = np.linspace(0,np.max(h_arr),11) # depth contour level
        fs = 12   # fontsize
        ls = 10   # labelsize

        # Plot
        fig = plt.figure(figsize = (20,8))
        gs = GridSpec(3,6)

        ss1 = gs.new_subplotspec((0,0),rowspan = 2,colspan = 2)  # contour
        ss2 = gs.new_subplotspec((0,2),colspan = 2)              # cross section
        ss3 = gs.new_subplotspec((1,2),colspan = 2)              # radius (taper)
        ss4 = gs.new_subplotspec((2,2),colspan = 2)              # index of super ellipse
        ss6 = gs.new_subplotspec((2,0),colspan = 2)              # information (just text)
        ss7 = gs.new_subplotspec((0,4),colspan = 2)              # area fraction
        ss8 = gs.new_subplotspec((1,4),colspan = 2)              # refractive index profile
        ss9 = gs.new_subplotspec((2,4),colspan = 2)              # transmission

        ax1 = plt.subplot(ss1) # contour
        ax2 = plt.subplot(ss2) # cross section
        ax3 = plt.subplot(ss3) # radius (taper)
        ax4 = plt.subplot(ss4) # index of super ellipse
        ax7 = plt.subplot(ss7) # area fraction
        ax8 = plt.subplot(ss8) # refractive index profile
        ax9 = plt.subplot(ss9) # transmission
        ax6 = plt.subplot(ss6) # information

        
        # =========================================
        # Contour
        # - - - - - - - - - - - - - - - - - - - - -
        ctlr = ax1.contour(X,Y,z_offset,levels =clevel, cmap = 'jet_r')
        ax1.clabel(ctlr)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X [mm]',fontsize = fs)
        ax1.set_ylabel('Y [mm]',fontsize = fs)
        ax1.tick_params(labelsize = ls)
        ax1.plot(x,np.zeros(len(x)),'r--')
        ax1.plot(np.zeros(len(x)),x,'b--')
        # - - - - - - - - - - - - - - - - - - - - -
        # =========================================

        
        # =========================================
        # Cross section
        # - - - - - - - - - - - - - - - - - - - - -
        ax2.plot(x,csx,'r',label = r'Cross section $x,\ d_x=%.3f$ mm'%(np.max(csx)-np.min(csx)))
        ax2.plot(x,csy,'b',label = r'Cross section $y,\ d_y=%.3f$ mm'%(np.max(csy)-np.min(csy)))
        for i in range(-2,3):
            ax2.plot(x+p*i,csx,'r')
            ax2.plot(x+p*i,csy,'b')
        ax2.tick_params(labelsize = ls)
        ax2.set_ylim(3.05,-0.05)
        ax2.grid()
        ax2.set_xlabel('Length [mm]',fontsize = fs)
        ax2.set_ylabel('Depth [mm]',fontsize = fs)
        ax2.legend(fontsize = ls)
        # - - - - - - - - - - - - - - - - - - - - -
        # =========================================

        
        # =========================================
        # Radius (taper)
        # - - - - - - - - - - - - - - - - - - - - -        
        ax3.plot(h_arr,r_x_taper[::-1],'r-',label = '$r_x(z)$')
        ax3.plot(h_arr,r_y_taper[::-1],'b-',label = '$r_y(z)$')
        ax3.grid()
        ax3.set_xlabel('Depth [mm]',fontsize = fs)
        ax3.set_ylabel('Radius [mm]',fontsize = fs)
        ax3.tick_params(labelsize = ls)
        ax3.legend(fontsize = ls)
        # - - - - - - - - - - - - - - - - - - - - -  
        # =========================================

        # =========================================
        # index of super ellipse
        # - - - - - - - - - - - - - - - - - - - - -
        ax4.plot(h_arr,nx_arr[::-1],'r-',label = '$n_x(z)$')
        ax4.plot(h_arr,ny_arr[::-1],'b-',label = '$n_y(z)$')
        ax4.grid()
        ax4.set_xlabel('Depth [mm]',fontsize = fs)
        ax4.set_ylabel('Index',fontsize = fs)
        ax4.tick_params(labelsize = ls)
        ax4.legend(fontsize = ls)
        # - - - - - - - - - - - - - - - - - - - - - 
        # =========================================
        
        # =========================================
        # area fraction
        # - - - - - - - - - - - - - - - - - - - - -        
        ax7.plot(h_arr,afrac[::-1], 'b-')
        ax7.grid()
        ax7.set_xlabel('Depth [mm]',fontsize = fs)
        ax7.set_ylabel('Area fraction',fontsize = fs)
        ax7.tick_params(labelsize = ls)
        # - - - - - - - - - - - - - - - - - - - - - 
        # =========================================        
        
        
        # =========================================
        # refractive index profile
        # - - - - - - - - - - - - - - - - - - - - -          
        ax8.plot(h_arr,n_emt[1][::-1], 'b-',label = '$n_{eff,~SWS}$')
        ax8.plot(z_klop,n_klop[1:],'k-',label = '$n_{Klop}$')
        ax8.grid()
        ax8.set_xlabel('Depth [mm]',fontsize = fs)
        ax8.set_ylabel(r'$n_{eff}$',fontsize = fs)
        ax8.tick_params(labelsize = ls)
        ax8.legend(fontsize = ls)
        # - - - - - - - - - - - - - - - - - - - - - 
        # =========================================

        # =========================================
        # transmission
        # - - - - - - - - - - - - - - - - - - - - -          
        ax9.plot(freq,trans,'b',label = '$T_{SWS}$')
        ax9.plot(freq,trans_klop,'k-',label = '$T_{Klop}$')
        ax9.fill_between([self.vb[0]*1e-9,self.vu[0]*1e-9],y1 = [0.0,0.0],y2=[2.0,2.0],color = 'orange',alpha = 0.3)
        ax9.fill_between([self.vb[1]*1e-9,self.vu[1]*1e-9],y1 = [0.0,0.0],y2=[2.0,2.0],color = 'orange',alpha = 0.3)
        ax9.text(self.vc[0]*1e-9,0.81,'$T_{ave,SWS}=%.3f$'%(np.mean(trans[self.band90_index])), ha = 'center')
        ax9.text(self.vc[1]*1e-9,0.81,'$T_{ave,SWS}=%.3f$'%(np.mean(trans[self.band150_index])), ha = 'center')
        ax9.grid()
        ax9.legend(fontsize = ls)
        ax9.set_xlabel('Frequency [GHz]', fontsize = fs)
        ax9.set_ylabel('Transmittance',fontsize = fs)
        ax9.tick_params(labelsize = ls)
        ax9.set_ylim(0.8,1.02)
        # - - - - - - - - - - - - - - - - - - - - - 
        # =========================================
                                
        # =========================================
        # information
        # - - - - - - - - - - - - - - - - - - - - -                                   
        x_text_off = 0.0
        ax6.set_xlim(0,1)
        ax6.text(x_text_off,1,r'$x,\ y,\ z$ resolution: {0}, {1}, {2}'.format(self.xy_ind,self.xy_ind,self.z_res))
        ax6.text(x_text_off,0.9, r'$p_x,\ p_y\ [mm] = {0}, {1}$'.format(p,p))
        ax6.text(x_text_off,0.8, r'$d_x,\ d_y,\ d_t\ [mm] = {0:.2f}, {1:.2f}, {2:.2f}$'.format(np.max(csx)-np.min(csx),np.max(csy)-np.min(csy),np.max(Z)-np.min(Z) ))
        ax6.text(x_text_off,0.7, r'$r_{x,top},\ r_{y,top}\ [mm]$'+' = {0}, {1}'.format(r_x_top,r_y_top))
        ax6.text(x_text_off,0.6, r'$r_{x,bottom},\ r_{y,bottom}\ [mm]$'+' = {0:.3f}, {1:.3f}'.format(r_x_bottom,r_y_bottom))
        ax6.text(x_text_off,0.5, r'$\alpha_x,\ \alpha_y = {0},{1}$'.format(alpha_x,alpha_y))
        ax6.text(x_text_off,0.4, r'$n_x[top, bottom],\ n_y[top, bottom] = {0},\ {1}$'.format(nx[::-1],ny[::-1]))
        ax6.text(x_text_off,0.3, r'$\beta_{x},\ \beta_{y}$'+ ' = {0},{1}'.format(beta_x, beta_y))
        # - - - - - - - - - - - - - - - - - - - - - 
        # =========================================
        

        # =========================================
        # 3D image
        # - - - - - - - - - - - - - - - - - - - - -          
        ax5 = plt.axes([.0, -.085, .5, .6],projection = '3d')
        ax5.set_xlim(-1.05,1.05)
        ax5.set_ylim(-1.05,1.05)
        ax5.set_zlim(2.05,-0.05)
        ax5.plot_surface(X,Y,z_offset, cmap='jet_r', antialiased=False,rcount=101,ccount=101)
        plt.axis('off')
        # - - - - - - - - - - - - - - - - - - - - - 
        # =========================================
        
        plt.savefig(lib.fig_path+f_save + '_geometry.png',dpi = 300, transparent = True) # save figure
        
        return X,Y,z_offset
    
