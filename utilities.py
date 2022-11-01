import os
import numpy as np
from numpy.testing import assert_almost_equal
import netCDF4 as nc
from IPython.display import clear_output
import matplotlib.pylab as plt
from mpl_toolkits import mplot3d
import pandas as pd
import requests
import shutil
import math
from sympy import total_degree


def acc_surf_rf(xx='R', latlon=None):
    # Example use: a = acc_surf_rf( latlon=(lat,lon) )
    len = range(8)
    XX = xx
    lat, lon = latlon
    dict = {"broadleaf trees": None,
            "needleleaf trees": None,
            "C3 temperate grass": None,
            "C4 temperate grass": None,
            "shrubs": None,
            "urban": None,
            "inland water:": None,
            "bare soil": None,
            "ice": None}
    key = list(dict)
    for i in len:
        csv = pd.read_csv(
            "/Users/cong/iCloud/Desktop/Monash/FYP/surface/surface_" + XX + "_" + str(i) + ".csv").to_numpy()
        a = csv[lat][lon]
        dict[key[i]] = a

    return dict


def get_lat_lon_index(wind_farm_coord):
    """
    This function was inspired by Qinghong Shi, he wrote the code for arranging lat and lon
    :param ds: the dataset
    :return:
    x_index: the index of matched latitude with desired wind farm in coordinate array
    y_index: the index of matched longitude in coordinate array
    """

    # The current range of longitude and latitude of nc file
    lat_max = 19.479996
    lat_min = -65
    lat_resolution = 0.10999298

    lon_max = 196.945
    lon_min = 65.055
    lon_resolution = 0.10998535

    # Equally space the lat and lon w.r.t the resolution
    x = np.arange(lat_min, lat_max, lat_resolution)
    y = np.arange(lon_min, lon_max, lon_resolution)

    my_coord = wind_farm_coord
    lat_index = 0
    lon_index = 0

    # loop through all the coordinates in dataset coordinates, find the difference lower than the resolution
    for i in range(len(x)):

        if abs(x[i] - my_coord[0]) < lat_resolution:
            lat_index = i
            break
    for j in range(len(y)):

        if abs(y[j] - my_coord[1]) < lon_resolution:
            lon_index = j
            break
    return lat_index, lon_index


def get_avg_data_specific(pm, base_dir, parameter: str, coordinates, mdl=None):
    # np.sort() will sort the files in ascending order, .tolist() because somehow, the ndarray can not hold too much str
    files = np.sort(os.listdir(base_dir)).tolist()

    for f in range(len(files)):
        files[f] = os.path.join(base_dir, files[f])

    # Create a holder for bunch of data
    total_data = []
    num_coord = coordinates.shape[0]
    total_data = np.zeros((num_coord, len(files) * 24))

    z = 0
    l = 0
    if f[-3:] == 'npz':
        total_data = np.zeros((num_coord, len(files) * 24))
    else:
        total_data = np.zeros((num_coord, len(files) * 6))
    for i in range(len(files)):
        f = files[i]
        print('current file', f)
        # Check if the file name start with parameter name, e.g if need fric_vel, file name:
        # fric_vel-fc-slv-PT1H-BARRA_R-v1-20150810T1800Z.sub.nc
        if parameter in f:

            if f[-3:] == 'npz':
                file = np.load(f)

                cmp = file[pm]
                file.close()
                for j in range(cmp.shape[0]):  # 24 hours
                    for c in range(num_coord):  # 9 coordinates
                        lat, lon = coordinates[c]
                        total_data[c][z] = cmp[j][mdl][lat][lon]
                    z += 1
                    if z % 50 == 0:
                        print("number hours: ", z)



            else:

                ds = nc.Dataset(f)
                for j in range(5):
                    for c in range(num_coord):  # 9 coordinates
                        lat, lon = coordinates[c]
                        total_data[c][l] = ds.variables[parameter][j][mdl][lat][lon]
                    l += 1
                    if l % 50 == 0:
                        print("number hours: ", l)
                ds.close()
    clear_output(True)
    # print('shape of all ',parameter, 'data: ',total_data.shape)

    # print('shape of averaged ',parameter, 'data: ',avg_data.shape)

    return total_data

def check_big_jump_based_on_time(file):
    
    from datetime import datetime, timedelta

    check_flag = False

    datetime_str = file[0]

    datetime_object = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')

    for i in range(file.shape[0]-1):

        datetime_str = file[i]

        datetime_object = datetime.strptime(datetime_str, '%Y/%m/%d %H:%M:%S')

        datetime_str2 = file[i+1]

        datetime_object2 = datetime.strptime(datetime_str2, '%Y/%m/%d %H:%M:%S')

        if (datetime_object2.minute - datetime_object.minute) != 5:
            if (datetime_object2.hour - datetime_object.hour) == 1 or 23:
                pass
            else:
                check_flag = True
                print('There is a big jump at indx:', i)
                print(f'The current index: {i}, time is {datetime_object}\nThe next time is {datetime_object2}\n')
    
    if check_flag == False:
        print('Check finished, no big jump in file.')
    else:
        print('There is a big jump at indicated index, please check carefully.')

class wind_speed_calc():
    def __init__(self) -> None:
        pass

    def wind_power_law(self, speed_ref, height_ref, height_turbine):
        """

        :param speed_r: wind speed at reference height. E.g wind speed at 10m or sea level, in this FYP it is 10m wind speed
        :param height_r: reference height. E.g wind speed at 10m height
        :param height_turbine: The center of turbine hub height w.r.t ground
        :return:
        x: The wind speed at turbine hub
        """

        # wind power law
        x = speed_ref * np.power((height_turbine / height_ref), 0.143)
        return x

    def wind_log_profile(self, targ_h, u0):
        # This log profile is without considering friction velocity, check Wiki: log profile
        # The surface roughness is considered as 2. 2 is generally for buildings in city, so it should be close enough to wind farm case
        uz = u0 * (np.log(targ_h / 0.25) / np.log(276.7 / 0.25))
        return uz


class BARRA_files():
    """
    root is something like: '/Volumes/New Volume/FYP/ucmp/2015/'
    """

    def __init__(self, parameters, root):
        self.parameters = parameters
        self.root = root

    def download(self):
        """
        The method will download a BARRA with specific set of parameters.
        ["R or PH, AD",
            "forecast or metadata",
            "mdl, prs, slv, spec or cld",
            "param: wnd_ucmp",
            "YYYY",
            "MM",
            "DD"]
        :return: None
        """
        parameters = self.parameters
        server = 'https://dapds00.nci.org.au/thredds/fileServer/cj37/BARRA/BARRA_'
        xx = parameters[0]
        forecast = parameters[1]
        mdl = parameters[2]
        parameter = parameters[3]
        year = parameters[4]
        month = parameters[5]
        day = parameters[6]
        durantion = ['00', '06', '12', '18']

        if day <= 9:
            day = "0" + str(day)
        else:
            day = str(day)
        for d in durantion:
            fname = parameter + '-fc-' + mdl + '-PT1H-BARRA_' + xx + '-v1-' + year + month + day + 'T' + d + '00Z.sub.nc'
            print(fname)
            link = server + xx + '/v1/' + forecast + mdl + '/' + parameter + '/' + year + '/' + month + '/' + fname
            print(link)
            r = requests.get(link, allow_redirects=True)
            with open(fname, 'wb') as f:
                f.write(r.content)
            # Move files to destination
            # shutil.move(fname, dst=self.root)

    def fcombine(self, specific_days=False, day=None, specific_month=False, s_month=None):
        import calendar

        # Define destination of root to save, and some file name
        root = self.root
        parameters = self.parameters
        xx = parameters[0]
        mdl = parameters[2]
        parameter = self.parameters[3]
        year = parameters[4]
        month = parameters[5]

        # If specific days has been defines, such as program runs from middle
        if specific_days:
            day = day
        elif specific_month:
            month = s_month
        else:
            # Define how many days in current month
            tmp = calendar.monthrange(int(year), int(month))[1]
            day = range(1,tmp+1)

        # each day
        for i in day:
            
            if i <= 9:
                i = "0" + str(i)
            date = year + month + str(i) + 'T'
            f1 = root + parameter + '-fc-' + mdl + '-PT1H-BARRA_' + xx + '-v1-' + date + '0000Z.sub.nc'
            f2 = root + parameter + '-fc-' + mdl + '-PT1H-BARRA_' + xx + '-v1-' + date + '0600Z.sub.nc'
            f3 = root + parameter + '-fc-' + mdl + '-PT1H-BARRA_' + xx + '-v1-' + date + '1200Z.sub.nc'
            f4 = root + parameter + '-fc-' + mdl + '-PT1H-BARRA_' + xx + '-v1-' + date + '1800Z.sub.nc'

            nc1 = nc.Dataset(f1)
            nc2 = nc.Dataset(f2)
            nc3 = nc.Dataset(f3)
            nc4 = nc.Dataset(f4)

            var1 = nc1.variables[parameter][:, :14, :, :]
            var2 = nc2.variables[parameter][:, :14, :, :]
            var3 = nc3.variables[parameter][:, :14, :, :]
            var4 = nc4.variables[parameter][:, :14, :, :]
            nc1.close()
            nc2.close()
            nc3.close()
            nc4.close()

            tmp = np.vstack((var1, var2, var3, var4))
            print(date)
            print(root + parameter + '-fc-mdl-PT1H-BARRA_R-v1-' + date + '0000Z.sub.nc')

            np.savez_compressed(root + parameter + '-fc-mdl-PT1H-BARRA_R-v1-' + date, parameter=tmp)
           # os.remove(f1)
           # os.remove(f2)
           # os.remove(f3)
           # os.remove(f4)
            clear_output(True)

    def get_latlon_data(self, coordinates, mdl_th=6):
    
    # mdl_th is which mdl height to choose, can referecence from below:
    # [ 1  2  3  4  5  6  7  9 11 
    # 13 15 17 19 21 23 25 27 29 
    # 31 33 35 37 39 41 43 45 47 
    # 49 51 53 55 57 59 61 63 65 67 69]
    
    # model Height (m): 
    # 10    36    76   130   196   276   370   596   876
    # 1210  1596  2036 2530  3076  3676  4330  5036  5796  
    # 6610  7476  8396  9371 10402 11493 12654 13898 15248 
    # 16742 18431 20392 22727 25580 29135 33637 39396 46807 56359 68660

        parameter = self.parameters[3]
        base_dir = self.root
        z = 0
        l = 0

        files = np.sort(os.listdir(base_dir)).tolist()
        for f in range(len(files)):
            files[f] = os.path.join(base_dir, files[f])

        num_coord = coordinates.shape[0]
        if files[15][
           -3:] == 'npz':  # Random chose a file name, if it is a npz file, then create m_coord * len(files) array
            total_data = np.zeros((num_coord, len(files) * 24))
        else:
            total_data = np.zeros((num_coord, len(files) * 6))
        for i in range(len(files)):
            f = files[i]
            print('current file', f)
            # Check if the file name start with parameter name, e.g if need fric_vel, file name:
            # fric_vel-fc-slv-PT1H-BARRA_R-v1-20150810T1800Z.sub.nc
            if parameter in f:
                if f[-3:] == 'npz':

                    # Load parameters in npy file
                    file = np.load(f)

                    # This will load what ever parameter in npz, such as wnd_ucmp or vcmp
                    cmp = file[file.files[0]]
                    file.close()

                    for j in range(cmp.shape[0]):  # 24 hours
                        for c in range(num_coord):  # 9 coordinates
                            lat, lon = coordinates[c]
                            total_data[c][z] = cmp[j][mdl_th][lat][lon]
                        z += 1
                        if z % 50 == 0:
                            print("number hours: ", z)

                else:  # f is nc file

                    ds = nc.Dataset(f)
                    for j in range(5):
                        for c in range(num_coord):  # 9 coordinates
                            lat, lon = coordinates[c]
                            total_data[c][l] = ds.variables[parameter][j][mdl_th][lat][lon]
                        l += 1
                        if l % 50 == 0:
                            print("number hours: ", l)
                    ds.close()

        clear_output(True)
        return total_data


class wind_farm_output(wind_speed_calc):
    """
    This class will compute the output of windfarm
    
    """

    def __init__(self, topography, ucmp, vcmp, WFname):
        wind_speed_calc.__init__(self)

        self.topography = topography
        self.ucmp = ucmp
        self.vcmp = vcmp
        self.WFname = WFname

    def wd_sp_hub(self):
        """
        This function takes the data of shape 720,1 which is one month duration
        """
        topography = self.topography
        hub_h = 83.5 + topography
        h = np.linspace(10, 300, 100) + topography
        # hub_h = np.full(h.shape,83.5+topography) # Hub height
        # surface = np.full(h.shape,topography) # surface height w.r.t sea level
        hub_vel_idx = np.where(np.logical_and(math.floor(hub_h) < h, h < math.ceil(hub_h)))[0]

        # velocity data shape: day, time, velocity
        u = self.ucmp.reshape(-1, 1)  # shape: 30*24*1, 1
        v = self.vcmp.reshape(-1, 1)  # shape: 30*24*1, 1
        vel = np.sqrt(u ** 2 + v ** 2)  # shape: 30*24*1, 1

        # Just make h to have a dim of (9, 100) for later calculating profile
        h1 = np.copy(h)
        for i in range(u.shape[0] - 1):
            h1 = np.vstack((h1, h))  # shape: 30*24*1, 300
        profile = self.wind_log_profile(targ_h=h1, u0=vel)  # shape: 30*24*1, 300

        vel_hub = np.zeros(profile.shape[0])
        for i in range(profile.shape[0]):
            vel_hub[i] = profile[i][hub_vel_idx]

        # Now this vel_hub has all the velocity at hub heigh, in one month, 24 hours
        vel_hub = vel_hub.reshape(30, 24, 1)
        return vel_hub

    def poweroutput(self, xlsx: str):

        # # Faltten the hub height_speed, previously it is days, hours,1
        # Now it is days * hours,1.
        vel_hub = self.wd_sp_hub().faltten()
        xlsx = pd.ExcelFile(xlsx)
        df = pd.read_excel(xlsx)
        # Find which row of WF is, such as Ararat WF, it is in row 3
        WF_indx = np.where(df["Wind Farm Name"] == self.WFname)[0][0]

        #  Retrieve how many turbines in WF
        number_of_turbines = pd.read_excel(xlsx, usecols='C').values[WF_indx]

        # Retrieve current WF turbine power curve
        power_curve = df.values[WF_indx]
        power_output = np.zeros(vel_hub.shape)

        # Then round values to nearest 0.5 to match the power curve
        wind_speed = np.round(vel_hub * 2) / 2
        for j in range(wind_speed.shape[0]):
            ws = wind_speed[j]
            power_output[j] = self.match_powercurve(ws, power_curve) * number_of_turbines[0]

        return power_output

    def match_powercurve(self, wind_speed, power_curve):
        for i in range(len(power_curve)):
            ws = power_curve[i].rsplit(",")[0].split('(')[1]  # Using split to get wind speed number. E.g. (4.5, 1839),
            # get 4.5
            ws = float(ws)  # The reason of converting to float is that some wind speed in excel is .5 eg. 1.5 2.5 4.5

            try:  # due to the float type cannot use normal comparison, we use assert_almost_equal() from library to
                # do the comparison between floats
                assert_almost_equal(ws, wind_speed)
            except AssertionError:
                continue
            else:

                power = power_curve[i].rsplit(",")[1].split(')')[0]
                power = int(power)
                return power

