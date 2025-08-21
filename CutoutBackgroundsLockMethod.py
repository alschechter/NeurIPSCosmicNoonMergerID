import MockImageFunctions
import astropy.io.fits as pyfits
import numpy as np
import pandas as pd
import sys
from glob import glob
import astropy.units as u
from astropy.wcs import WCS
import astropy.wcs as w
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import os 
from multiprocessing import Pool, Manager, Lock
import concurrent.futures
# import matplotlib
# matplotlib.use('Agg')
errorlog = open("cutoutbackgrounds_poolparallel50.log", "w")

# try:
#     oldfiles = glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/*.fits')
#     for old in oldfiles:
#         os.remove(old)
#     os.remove('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/RADec40.txt')
#     oldfiles = glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/*.fits')
#     print('length of directory currently', len(oldfiles))
#     print('removed old fits files')
# except OSError as error: 
#     print(error) 
#     print("File path can not be removed") 

try:
    oldfiles = glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/z1noRT/*.fits')
    for old in oldfiles:
        os.remove(old)
    os.remove('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/z1noRT/RADec50.txt')
    oldfiles = glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/z1noRT/*.fits')
    print('length of directory currently', len(oldfiles))
    print('removed old fits files')
except OSError as error: 
    print(error) 
    print("File path can not be removed") 
    
    
# try:
#     oldfiles = glob('testmocks/*.fits')
#     for old in oldfiles:
#         os.remove(old)
#     os.remove('testmocks/RADec84.txt')
#     oldfiles = glob('testmocks/*.fits')
#     print('length of directory currently', len(oldfiles))
#     print('removed old fits files')
# except OSError as error: 
#     print(error) 
#     print("File path can not be removed") 
    
print(len(pd.read_csv('SubhaloListForMakeMocks50.csv').index))

mosaic_path_125 = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/mosaics/mosaics/hlsp_candels_hst_wfc3_cos-tot_f125w_v1.0_drz.fits'
mosaic_path_606 = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/mosaics/mosaics/hlsp_candels_hst_acs_cos-tot_f606w_v1.0_drz.fits'
mosaic_path_160 = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/mosaics/mosaics/hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits'
mosaic_path_814 = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/mosaics/mosaics/hlsp_candels_hst_acs_cos-tot_f814w_v1.0_drz.fits'

mosaic_125 = pyfits.open(mosaic_path_125)
mosaic_606 = pyfits.open(mosaic_path_606)
mosaic_160 = pyfits.open(mosaic_path_160)
mosaic_814 = pyfits.open(mosaic_path_814)

mosaicdata_125 = mosaic_125[0].data
mosaicdata_606 = mosaic_606[0].data
mosaicdata_160 = mosaic_160[0].data
mosaicdata_814 = mosaic_814[0].data
# if np.all(mosaicdata_125.data == 0) or np.all(mosaicdata_160.data == 0) or np.all(mosaicdata_606.data == 0) or np.all(mosaicdata_814.data == 0):
#     print('mosaic all0s')
# else:
#     print('mosaic means', np.mean(mosaicdata_125), np.mean(mosaicdata_606), np.mean(mosaicdata_160), np.mean(mosaicdata_814))

#print(np.shape(mosaicdata))
mosaicheader_125 = mosaic_125[0].header
mosaicheader_606 = mosaic_606[0].header
mosaicheader_160 = mosaic_160[0].header
mosaicheader_814 = mosaic_814[0].header

#print(mosaicheader_125)
# if np.all(mosaicdata_125.data == 0) or np.all(mosaicdata_160.data == 0) or np.all(mosaicdata_606.data == 0) or np.all(mosaicdata_814.data == 0):
#     print('all0s')
# else: 
#     print(np.mean(mosaicdata_125), np.mean(mosaicdata_160), np.mean(mosaicdata_606), np.mean(mosaicdata_814))    
# exit()   
del mosaicheader_125['A_*']; del mosaicheader_125['B_*']
del mosaicheader_160['A_*']; del mosaicheader_160['B_*']
del mosaicheader_606['A_*']; del mosaicheader_606['B_*']
del mosaicheader_814['A_*']; del mosaicheader_814['B_*']

wcsheader_125 = w.WCS(mosaicheader_125)
wcsheader_125.sip = None    
wcsheader_606 = w.WCS(mosaicheader_606)
wcsheader_606.sip = None    
wcsheader_160 = w.WCS(mosaicheader_160)
wcsheader_160.sip = None  
wcsheader_814 = w.WCS(mosaicheader_814)
wcsheader_814.sip = None  
print('at bad list')

badlist50 = MockImageFunctions.MakeBadList('/n/holystore01/LABS/hernquist_lab/Users/aschechter/mosaics/mosaics/hlsp_candels_hst_wfc3_cos-tot_f125w_v1.0_drz.fits',\
                                         '/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/CANDELS_RAdecOnly.csv')


badlist = np.loadtxt('badlist50.txt')
print(np.shape(badlist))
#goodlist = np.loadtxt('goodlist_extrasave.txt')
#goodlist = MockImageFunctions.PerformCutout('/n/holystore01/LABS/hernquist_lab/Users/aschechter/mosaics/mosaics/hlsp_candels_hst_wfc3_cos-tot_f125w_v1.0_drz.fits', badlist, 'wfc3_ir_f125w', 0.2)
#print(len(goodlist))

RADec_list2 = []
failed_center = []
RADec_list2_arraytest = np.array([])
failed_center_arraytest = np.array([])
z = 1 #change to what redshift we want
kpc_per_side = 100*((1+z)**(-1))* u.kpc
camera_pixel_scale_acs_wfc3_uvis = 0.03 * u.arcsec
camera_pixel_scale_wfc3_ir = 0.06 * u.arcsec
kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec)
arcsec_per_side = kpc_per_side / kpc_per_arcsec
pixels_per_side_acs_wfc3_uvis = arcsec_per_side / camera_pixel_scale_acs_wfc3_uvis
pixels_per_side_wfc3_ir = arcsec_per_side / camera_pixel_scale_wfc3_ir 

#print(pixels_per_side_acs_wfc3_uvis, pixels_per_side_wfc3_ir)

goodlist = []
goodlist_coords = []
goodlist_coords_use = []
RADec_list = []

numberneeded_array = np.arange(0,len(glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/fullmocks/z1/*f125w.npy')) + 50) #sometimes it fails and we need extra

print(len(numberneeded_array))
#numberneeded_array = np.arange(0,4)
output_file = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/RADec50.txt'

def bigfunction(numberneeded, output_file, lock, shared_list):   #lock suggested by chatGPT
    if numberneeded % 10 == 0:
        print(numberneeded)
    #lock = Lock()
    coords_useable = False 
    writing_file_successfully = False
    counter = 0
    #while len(glob('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/wfc3_ir_f125w*.fits')) <= numberneeded: #change this to how many we need
    # used to be:len(pd.read_csv('SubhaloListForMakeMocks84.csv'))
    while writing_file_successfully == False:
        while coords_useable == False:
            print('attempt # ', counter+1)
            #changing from np.random.randint(0,int(np.shape(mosaicdata_125)[0]+1)) to avoid edge
            random_RA = np.random.randint(100,int(np.shape(mosaicdata_125)[1]-100))
            random_dec = np.random.randint(100,int(np.shape(mosaicdata_125)[0]-100))
            # print('mosiac length', np.shape(mosaicdata_125)[1], np.shape(mosaicdata_125)[0])
            # print('random_RA', random_RA)
            # print('random dec', random_dec)
            random_center = [random_RA, random_dec]
            #print(np.shape(random_center))
            random_center_coords = np.array(wcsheader_125.pixel_to_world_values(random_center[0], random_center[1]))
            print('random center: ', random_center, random_center_coords)
            if random_center in badlist:
                print('source overlap! in bad')
                counter += 1
                coords_useable = False
                # random_RA = np.random.randint(100,int(np.shape(mosaicdata_125)[1]-100))
                # random_dec = np.random.randint(100,int(np.shape(mosaicdata_125)[0]-100))
                # random_center = [random_RA, random_dec]
                # random_center_coords = np.array(wcsheader_125.pixel_to_world_values(random_center[0], random_center[1]))
                print('random center: ', random_center, random_center_coords)
            elif random_center in goodlist:
                print('source overlap! in good')
                counter += 1
                coords_useable = False
                # random_RA = np.random.randint(100,int(np.shape(mosaicdata_125)[1]-100))
                # random_dec = np.random.randint(100,int(np.shape(mosaicdata_125)[0]-100))
                # random_center = [random_RA, random_dec]
                # random_center_coords = np.array(wcsheader_125.pixel_to_world_values(random_center[0], random_center[1]))
                print('random center: ', random_center, random_center_coords)

            else:
                coords_useable = True
                print('attempting cutout', random_center)
                
        goodlist.append(random_center)
        goodlist_coords.append(random_center_coords)

        # for n in range(len(goodlist)):
        #     n = int(sys.argv[1])
        #     m = n-1
            #m = n+1
        #for m in range(2):
        coordpair = SkyCoord(random_center_coords[0], random_center_coords[1], unit="deg")
        #print('coordpair', coordpair)
        cutout_125 = Cutout2D(data = mosaicdata_125, position = coordpair, size = pixels_per_side_wfc3_ir.value, 
                    wcs = wcsheader_125, fill_value= 0, copy = True)
        #print(cutout_125)
        cutout_160 = Cutout2D(data = mosaicdata_160, position = coordpair, size = pixels_per_side_wfc3_ir.value, 
                    wcs = wcsheader_160, fill_value= 0, copy = True)
        cutout_606 = Cutout2D(data = mosaicdata_606, position = coordpair, size = pixels_per_side_acs_wfc3_uvis.value, 
                    wcs = wcsheader_606, fill_value= 0, copy = True)
        cutout_814 = Cutout2D(data = mosaicdata_814, position = coordpair, size = pixels_per_side_acs_wfc3_uvis.value, 
                    wcs = wcsheader_814, fill_value= 0, copy = True)
        # print('shape test', np.shape(cutout_125.data)[0])
        # if np.mean(cutout_814.data) == 0.:
        #     print('0 mean 814')
        #     continue
        # elif np.all(cutout_160.data == 0):
        #     print('all0s 160')
        #     continue
        # elif np.all(cutout_606.data == 0):
        #     print('all0s 606')
        #     continue
        # elif np.all(cutout_814.data == 0):
        #     print('all0s 814')
        #     continue
        #trying without this condition to see if it's causing problems
        if cutout_125.data.all() != 0. and cutout_160.data.all() != 0. and cutout_606.data.all() != 0. and cutout_814.data.all() != 0.:    
            try:
                with lock: 
                    #photflam_125 = mosaic_125[0].header['PHOTFLAM']
                    #print(photflam_125)
                    #data125 = cutout_125*photflam_125
                    #np.savetxt('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/wfc3_ir_f125w' + str(np.round(coordpair.ra.value, decimals = 10)) + '_' + str(np.round(coordpair.dec.value, decimals = 10)) + '.npy', data_125)
                        
                    photflam_125 = mosaic_125[0].header['PHOTFLAM']
                    hdu_125 = pyfits.PrimaryHDU(cutout_125.data*photflam_125)
                    hdulist_125 = pyfits.HDUList([hdu_125])
                    #print(hdulist_125)
                    
                    photflam_606 = mosaic_606[0].header['PHOTFLAM']
                    hdu_606 = pyfits.PrimaryHDU(cutout_606.data*photflam_606)
                    hdulist_606 = pyfits.HDUList([hdu_606])
                    #print(hdulist_606)
                    #print('606', np.shape(hdulist_606), type(hdulist_606))
                    
                    photflam_160 = mosaic_160[0].header['PHOTFLAM']
                    hdu_160 = pyfits.PrimaryHDU(cutout_160.data*photflam_160)
                    hdulist_160 = pyfits.HDUList([hdu_160])
                    #print(hdulist_160)
                    #print('160', np.shape(hdulist_160), type(hdulist_160))
                    
                    photflam_814 = mosaic_814[0].header['PHOTFLAM']
                    hdu_814 = pyfits.PrimaryHDU(cutout_814.data*photflam_814)
                    hdulist_814 = pyfits.HDUList([hdu_814])
                    #print(hdulist_814)
                    #print('814', np.shape(hdulist_814), type(hdulist_814))
                    
                    print('cutout created')
                    ### adding paths in a better way lol
                    coordpair_str = f"{MockImageFunctions.truncate(coordpair.ra.value, 10)}_{MockImageFunctions.truncate(coordpair.dec.value, 10)}"
                    output_dir = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/z1noRT/'
                    hdulist_125.writeto(f"{output_dir}wfc3_ir_f125w_{coordpair_str}.fits")
                    hdulist_606.writeto(f"{output_dir}acs_wfc_f606w_{coordpair_str}.fits")
                    hdulist_160.writeto(f"{output_dir}wfc3_ir_f160w_{coordpair_str}.fits")
                    hdulist_814.writeto(f"{output_dir}acs_wfc_f814w_{coordpair_str}.fits")
        
                    
                    # Append the coordinate to shared list  from ChatGPT to not overwrite my own work
                    with open(output_file, "a") as f:
                        f.write(f"{coordpair_str}\n")

                    shared_list.append(coordpair_str)  # Append to the shared list

                    writing_file_successfully = True
                    return coordpair_str
                    #coordpair_str_towrite #str(MockImageFunctions.truncate(coordpair.ra.value, 10)) + '_' + str(MockImageFunctions.truncate(coordpair.dec.value, 10))

            except Exception as e:
                #exc_type, exc_obj, exc_tb = sys.exc_info() this made everything stop printing...
                print('failed to write files...', f"{str(e)}\n")
                #print('do cutouts fail?', np.shape(hdu_125), np.shape(hdu_606), np.shape(hdu_160), np.shape(hdu_814))
                # errorlog.write(str(e))
                #coords_useable = False
                return 'Fail'
        # else:
        #     continue

        else:
            print('this point does not work')
            #failed_center.append(str(MockImageFunctions.truncate(coordpair.ra.value, 10)) + '_' + str(MockImageFunctions.truncate(coordpair.dec.value, 10)))
            #print("Length of failed center array = ", len(failed_center))
            coords_useable = False
    #     #write line ti
   


# def insert_coords(coords, coords_list): 
#     #add to list
#     coords_list.append(coords) 


def main():
    
    print('range(len(numberneeded_array))', range(len(numberneeded_array)))
    
    # Initialize shared resources -- this helps so I can write with multiple workers
    with Manager() as manager:
        lock = manager.Lock()
        # Initialize shared resources
        shared_list = manager.list()
        output_file = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/RADec50.txt'

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(bigfunction, i, output_file, lock, shared_list): i for i in range(len(numberneeded_array))} ##range(len(numberneeded_array))
            
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    data = future.result()
                    #print('data', data)  # Optional: Print output to console
                except Exception as exc:
                    errorlog.write(f'Failed at Item {index} generated an exception: {exc}\n') 
                    # After all workers are done, write the full list of coordinates to a file
        with open(output_file, "a") as f:
            for coord in shared_list:
                f.write(f"{coord}\n")
        print("Finished writing all coordinates to file")        

if __name__ == "__main__":
    main()


# this function works for parallel, but doesn't write out correctly
# if __name__ == '__main__':

#     # create a pool of 4 worker processes
#     p = Pool(processes=4)
    
    
#     # assign bigfunction to a worker for each entry in len(numberneeded).
#     # each array entry is passed as an argument to sleep_func
#     p.map_async(bigfunction, numberneeded_array)
    
#     for coordpair in p.imap_unordered(bigfunction, numberneeded_array):
#         print(coordpair, file=output_file)
        
#     p.close()
#     p.join()
    
    
    # ##### NEED TO CHANGE TO APPEND TO A FILE INSTEAD OF ARRAY
    # np.savetxt('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/RADec84.txt', np.array(RADec_list2), fmt="%s")
    # np.savetxt('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/RADec84_arraytest.txt', RADec_list2_arraytest, fmt="%s")
    # np.save('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/RADec84_savetest.npy', np.array(RADec_list2))
    # np.savetxt('/n/holystore01/LABS/hernquist_lab/Users/aschechter/background_cutouts/failedRADec84.txt', np.array(failed_center), fmt="%s")