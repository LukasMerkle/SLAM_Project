# !/usr/bin/python
import numpy as np
import open3d as o3d
import sys
import struct
def visualize_cloud(cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    # frame = [o3d.geometry.TriangleMesh.create_coordinate_frame()]
    o3d.visualization.draw_geometries([pcd])

def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def verify_magic(s):
    
    magic = 44444

    m = struct.unpack('<HHHH', s)

    return len(m)>=4 and m[0] == magic and m[1] == magic and m[2] == magic and m[3] == magic

def main(args):
    
    if len(sys.argv) < 2:
        print ("Please specifiy input bin file")
        return 

    f_bin = open(sys.argv[1], "rb")

    total_hits = 0
    first_utime = -1
    last_utime = -1

    while True:

        magic = f_bin.read(8)
        if magic == '': # eof
            break

        if not verify_magic(magic):
            print ("Could not verify magic")

        num_hits = struct.unpack('<I', f_bin.read(4))[0]
        utime = struct.unpack('<Q', f_bin.read(8))[0]

        padding = f_bin.read(4) # padding

        print ("Have %d hits for utime %ld" % (num_hits, utime))

        total_hits += num_hits
        if first_utime == -1:
            first_utime = utime
        last_utime = utime
        current_cloud = np.zeros((0,3))
        print(num_hits)
        for i in range(num_hits):

            x = struct.unpack('<H', f_bin.read(2))[0]
            y = struct.unpack('<H', f_bin.read(2))[0]
            z = struct.unpack('<H', f_bin.read(2))[0]
            i = struct.unpack('B', f_bin.read(1))[0]
            l = struct.unpack('B', f_bin.read(1))[0]

            x, y, z = convert(x, y, z)
            s = "%5.3f, %5.3f, %5.3f, %d, %d" % (x, y, z, i, l)
            current_cloud = np.vstack((current_cloud,np.array([x,y,z])))
            # print (s)
        print(current_cloud.shape)
        visualize_cloud(current_cloud)
        # input("Press enter to continue...")

    f_bin.close()




if __name__ == '__main__':
    sys.exit(main(sys.argv))