import csv
from PIL import Image
from tflearn.data_utils import build_hdf5_image_dataset

class SideFunctions:
	def ExtractFromCSV(csv_in,output_dir,image_size,image_mode='RGB'):
		with open(csv_in,'r') as csvin:
			traindata=csv.reader(csvin, delimiter=',', quotechar='"')
			rowcount=0
			for row in traindata:
				if rowcount > 0:
					print('rows ' + str(rowcount) + "\n")
					x=0
					y=0
					pixels=row[1].split()
					img = Image.new(image_mode,image_size)
					for pixel in pixels:
						colour=(int(pixel),int(pixel),int(pixel))
						img.putpixel((x,y), colour)
						x+=1
						if x >= 48:
							x=0
							y+=1
					imgfile=output_dir+'/'+str(row[0])+'/'+str(rowcount)+'.png'
					img.save(imgfile,'png')
				rowcount+=1

	def BuildH5FromDirectory(directory,size):
		build_hdf5_image_dataset(directory, image_shape=size, mode='folder', grayscale= True, categorical_labels=True, normalize=True)

#SideFunctions.ExtractFromCSV("./Dataset/Split/validation.csv","./Dataset/Split/Validation/",(48,48))
SideFunctions.BuildH5FromDirectory("./Dataset/John/classified/",(48,48))
