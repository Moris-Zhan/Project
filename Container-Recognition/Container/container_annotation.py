import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
from os.path import join
import shutil



classes = ["water_tower","bowl","plate","bottle","bucket","washing_machine","plastic_bag",
"box","toilet","aquarium","tire","tub","styrofoam"]


def convert(size, box):
    image_w = size[0]
    image_h = size[1]

    dw = 1./(image_w)
    dh = 1./(image_h)
    x = (box[0] + (box[1] - box[0])/2.0) * 1.0 * dw
    y = (box[2] + (box[3] - box[2])/2.0) * 1.0 * dh
    w = (box[1] - box[0])*1.0 * dw
    h = (box[3] - box[2])*1.0 * dh    
    return (x, y, w, h)

def convert_annotation(image_id, label_file,train_file):
    in_file = open('./train_cdc/train_annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    width = int(root.find('size').findtext('width'))
    height = int(root.find('size').findtext('height'))

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        #xmlbox = obj.find('bndbox')

        '''b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        label_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
	label_file.write('\n')'''

        xmin = obj.find('bndbox').findtext('xmin')
        ymin = obj.find('bndbox').findtext('ymin')
        xmax = obj.find('bndbox').findtext('xmax')
        ymax = obj.find('bndbox').findtext('ymax')
        train_file.write( xmin + "," + ymin + "," + xmax + ","  + ymax + ","  + str(cls_id) + ' ')

        b = (float(xmin), float(xmax), float(ymin), float(ymax))
        bb = convert((width, height), b)
        label_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        print("name:{} xmin:{} ymin:{} xmax:{} ymax:{}".format(cls,bb[0],bb[1],bb[2],bb[3]))

    train_file.write('\n')

wd = getcwd()
if not os.path.exists('./JPEGImages/'):
    os.makedirs('./JPEGImages/')


if not os.path.exists('./train_cdc/train_labels/'):
    os.makedirs('./train_cdc/train_labels/')


train_XmlDir = "./train_cdc/train_annotations/"
train_file = open('container_train.txt','w') 
for idx,f in enumerate(listdir(train_XmlDir)):    
    image_id = f.replace(".xml","")   
    train_file.write(wd+'\JPEGImages\%s.jpg'%(image_id) + " ")
    label_file = open('./train_cdc/train_labels/%s.txt'%(image_id),'w')   
    #img_file = open('./train_cdc/train_images/%s.jpg'%(image_id),'w')  
    #shutil.copy2('./train_cdc/train_images/%s.jpg'%(image_id), './JPEGImages/') 
    shutil.copyfile('./train_cdc/train_images/%s.jpg'%(image_id), './JPEGImages/%s.jpg'%(image_id))

    convert_annotation(image_id, label_file,train_file)
    
    label_file.close()

print("Finish...")

