def displayImage(image,title):
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def displaysubImage(img,title,r,c,p): 
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    fig.add_subplot(r, c, p)
    plt.imshow(img,cmap=plt.cm.gray)
    plt.title(title)

def excludeChannel(image,channel):
    image[:,:,channel] = 0
    return image

def myConvolve2d(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image

    for x in range(image.shape[0]):     # Loop over every pixel of the image
        for y in range(image.shape[1]):
            # element-wise multiplication and summation 
            output[x,y]=(kernel*image_padded[x:x+3,y:y+3]).sum()
    return output
def histo(img1):
    hist,bins = np.histogram(img1.flatten(),256,[0,256])
    plt.hist(img1.flatten(),256,[0,256], color = 'b')
    plt.show()
    
def Gaussian_Noise(mean,var,sigma):
    gauss = np.random.normal(mean,sigma,image.shape)
    gauss = gauss.reshape(image.shape)
    noisy_image = image + gauss
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image
def MeshGauss(u,s):
    x=np.linspace(-10,10, num=100)
    y=np.linspace(-10,10, num=100)
    x, y = np.meshgrid(x, y)
    z = np.exp((-u*x**2-u*y**2))/s
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x,y,z, cmap=cm.jet)
    plt.show()
def SaltnPepper(image):
    rows,col,channels=image.shape
    pr = 0.05
    sp_img = np.zeros(image.shape, np.uint8)

    for i in range(rows):
        for j in range(col):
            r=random.random()
            if r < pr/2:
                sp_img[i][j] = [0, 0, 0]
            elif r < pr:
                sp_img[i][j] = [255, 255, 255]
            else:
                sp_img[i][j] = image[i][j]

    displayImage(sp_img,'Salt and Pepper Noise')
