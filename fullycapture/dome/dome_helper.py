import os
import glob
import fullycapture.data.dataIO as dataIO
def dome_skel_vga_to_hd(rootpath,vga_folder,hd_folder):
    #get frameids
    data_folder = os.path.join(rootpath,vga_folder)
    all_skels_vga = glob.glob(os.path.join(data_folder,'*.json'))
    start_id = dataIO.filepath_to_frameid(all_skels_vga[0])
    end_id = dataIO.filepath_to_frameid(all_skels_vga[-1])
    print(start_id)
    print(end_id)