from xml.etree.ElementTree import ElementTree
import  sys,os
import struct

def typeList():
    return {
        "52617221": EXT_RAR,
        "504B0304": EXT_ZIP,
       "3C3F786D6C": xml }

def bytes2hex(bytes):
    num = len(bytes)
    hexstr = u""
    for i in range(num):
        t = u"%x" % bytes[i]
        if len(t) % 2:
            hexstr += u"0"
        hexstr += t
    return hexstr.upper()


def filetype(filename):
    binfile = open(filename, 'rb')
    tl = typeList()
    ftype = 'unknown'
    for hcode in tl.keys():
        numOfBytes = len(hcode) / 2
        binfile.seek(0)
        hbytes = struct.unpack_from("B" * numOfBytes, binfile.read(numOfBytes))
        f_hcode = bytes2hex(hbytes)
        if f_hcode == hcode:
            ftype = tl[hcode]
            break
    binfile.close()
    print(ftype)
    return ftype

#encoding=utf-8
object_count = 0
xml_count = 0
image_count = 0
def countImage(image_dir):
    global image_count
    fs = os.listdir(image_dir)
    for f1 in fs:
        # print("in while")
        tmp_path = os.path.join(image_dir, f1)
        if not os.path.isdir(tmp_path):
            if tmp_path.split('.')[1] == 'jpg':
                image_count += 1
        else:
            print('dir: %s' % tmp_path)
            countImage(tmp_path)

def countObjectMain(xml_dir):
    if not os.path.isdir(xml_dir):
        global object_count
        global xml_count
        xml_count = 1
        object_count += count(xml_dir)
    else:
        countObject(xml_dir)

def countObject(xml_dir):
    #xml_dir=unicode(xml_dir,'utf-8')
    global object_count
    global xml_count
    fs = os.listdir(xml_dir)
    for f1 in fs:
        #print("in while")
        tmp_path = os.path.join(xml_dir, f1)
        if not os.path.isdir(tmp_path):
            if tmp_path.split('.')[1] == 'xml':
                xml_count += 1
                object_count += count(tmp_path)
        else:
            print('dir: %s' % tmp_path)
            countObject(tmp_path)

def count(filename):
    tree =ElementTree()
    tree.parse(filename)
    count = 0
    all_objects = tree.getroot().getchildren()

    for object in all_objects:
        if object.tag == "object":
            count = count +1
    return count

if __name__ == "__main__":
    countObjectMain(sys.argv[1])
    #countImage(sys.argv[2])
    #print(image_count)
    f = open('./result.txt', 'w')
    f.writelines("the number of object is : %d\n" %object_count)
    f.writelines("the number of xml(images) is : %d" %xml_count)
    f.close()
    print(object_count)
    print(xml_count)