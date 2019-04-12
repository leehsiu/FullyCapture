import fullycapture.dome.dome_helper as dome_helper
import os
if __name__=='__main__':
    rootpath = '/media/domedbweb/develop/webdata/dataset'
    seqname = '171204_pose6'
    vga_folder = 'vgaPose3d_stage1_op25'
    root_folder = os.path.join(rootpath,seqname)
    dome_helper.dome_skel_vga_to_hd(root_folder,vga_folder,vga_folder)
    