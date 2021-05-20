import imgaug.augmenters as iaa
import imgaug as ia
import cv2
import os


def load_image(folder_path):
    images = [ cv2.imread(os.path.join(folder_path,fn), cv2.IMREAD_COLOR) for fn in os.listdir(folder_path)]
    return images

def save_aug_images(images, init_index = 0, subdir='sub', name='None', process='None'):
    # dummy function, implement this
    
    if not os.path.exists(os.path.join('augd_img', subdir, name)):
        os.makedirs(os.path.join('augd_img', subdir, name))

    for img in images:
        cv2.imwrite(os.path.join('augd_img',subdir,name, process+str(init_index)+'.jpg'), img)
        init_index += 1

# Pipeline:
# (1) Crop images from each side by 1-16px, do not resize the results
#     images back to the input size. Keep them at the cropped size.
# (2) Horizontally flip 50% of the images.
# (3) Blur images using a gaussian kernel with sigma between 0.0 and 3.0.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.BlendAlphaSimplexNoise(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.BlendAlphaFrequencyNoise(
                        exponent=(-4, 0),
                        foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                        background=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)

seq_blur = iaa.Sequential([
    iaa.Crop(px=(1, 16), keep_size=False),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0))
])

seq_rot = iaa.Sequential([
    iaa.PerspectiveTransform(scale=(0.01, 0.15)),
    iaa.Rot90((1, 3))
])

seq_shear = iaa.Sequential([
    iaa.Crop(px=(1, 16), keep_size=False),
    iaa.ShearX((-20, 20)),
    iaa.ShearY((-20,20))
])

seq_scale = iaa.Sequential([
    iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})
])

seq_offset = iaa.Sequential([
    iaa.Affine(translate_px={"x": (-100, 100), "y": (-100, 100)}),
    iaa.Crop(px=(1, 16), keep_size=False)
])



def create_aug_img(folder_path):
    contents = folder_path.split('/')
    subdir = contents[1]
    name = contents[2]

    images = load_image(folder_path)
    images_aug = seq(images=images)
    save_aug_images(images_aug, init_index=0, subdir=subdir, name=name, process='complex')

    images = load_image(folder_path)
    images_aug = seq_blur(images=images)
    save_aug_images(images_aug, init_index=0, subdir=subdir, name=name, process='blur')

    images_aug = seq_rot(images=images)
    save_aug_images(images_aug, init_index=0, subdir=subdir, name=name, process='rotate')

    images_aug = seq_shear(images=images)
    save_aug_images(images_aug, init_index=0, subdir=subdir, name=name, process='shear')

    images_aug = seq_scale(images=images)
    save_aug_images(images_aug, init_index=0, subdir=subdir, name=name, process='scale')

    images_aug = seq_offset(images=images)
    save_aug_images(images_aug, init_index=0, subdir=subdir, name=name, process='offset')
if __name__ == '__main__':
    
    img_folders = ['seefood/train/hot_dog', 'seefood/test/hot_dog',
             'seefood/train/not_hot_dog', 'seefood/test/not_hot_dog']

    for folder in img_folders:
        create_aug_img(folder)
