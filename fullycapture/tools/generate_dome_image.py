import cv2
import os.path
import totaldensify.data.dataIO as dataIO

#set params here
seq_name = '170407_haggling_b3'
root_path = '/media/posefs0c/panoptic'
frame_range = range(3500,3600)
view_range = range(0,31)
dest_path = '/home/xiul/databag/dome_sptm'

calib_file = os.path.join(root_path,seq_name,'calibration_{}.json'.format(seq_name))

K,R,t,dist = dataIO.load_dome_calibs(calib_file,view_range)

for frameid in frame_range:
    for viewid in view_range:
        imgfile = os.path.join(root_path,seq_name,'hdImgs','00_{:02d}'.format(viewid),'00_{:02d}_{:08d}.jpg'.format(viewid,frameid))
        if os.path.isfile(imgfile):
            img = cv2.imread(imgfile)
            img_undist = cv2.undistort(img,K[viewid],dist[viewid])
            img_out = cv2.resize(img_undist,(1280,720))
            out_path = os.path.join(dest_path,seq_name,'images','00_{:02d}_{:08d}.jpg'.format(viewid,frameid))
            cv2.imwrite(out_path,img_out)
            print('[{}/{}:{}/{}]'.format(frameid,len(frame_range),viewid,len(view_range)))
