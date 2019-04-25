

'''
Author: Zhenxin Fu
Email: fuzhenxin95@gmail.com
Usage:
    python cal_p1.py your_output_file groundtruth_file
Example:
    python cal_p1.py valid/valid_sample_this_is_not_groundtruth.txt valid/valid_ground.txt
'''


import sys


def cal_p1(f_out, f_ground):
    lines_out = open(f_out, "r").readlines()
    lines_ground = open(f_ground, "r").readlines()
    lines_out = [int(i.strip()) if len(i.strip())!=0 else -1 for i in lines_out ]
    lines_ground = [int(i.strip()) for i in lines_ground]
    p1 = 0.
    for i in range(len(lines_ground)):
        res = lines_out[i*11]
        if res==lines_ground[i]:
            p1 += 1.
    return p1/len(lines_ground)



if __name__=="__main__":
    if len(sys.argv)!=3:
        print("python cal_p1.py sample.txt ground.txt")
    res = cal_p1(sys.argv[1], sys.argv[2])
    print(res)
