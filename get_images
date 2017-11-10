"""if the images are on you computers hardrive. Arguments are the files path and, 
name(path), the desired size of pixel array(*args), and optional argument for RGB = True"""

""" example - images = comp_images('/Users/bennicholl/Desktop/cars' , 110,110)"""
"""the example outputs the images of cars in a FOLDER on my compiuters
   desktop as a 110 X 110 greyscale pixel images"""

def comp_images(path, *args, add_fourth_d = True, greyscale = True):
    li = []
    for file in os.listdir(path):
        try:
            """read the pixel array from the picture in its respecitve path"""
            pic = io.imread(path + '/' + file, greyscale)
            """if resize arguments are True, resize images so they are all the same"""
            if len(args) > 0:
                pic = imresize(pic,args)
            li.append(pic)
        except:
            print('doesnt work')       
    return np.array(li)
