"""This package includes all transformations functions.

They are transformations-classes and can be concatenated
"""
import numpy as np
import numpy.ma as ma
import torch
import copy


#Transformations for the 3D approaches

## Normalization

class PercentileNormalize3D(object):
    """
        Percentile based normalization

    Attributes
    ----------
        mi : float
            lower limit used for the normalisation
        ma : float
            top limit used for the normalisation
        should_be_zero_one_interval : bool, optional
            flag to return the data being in a certain interval
            if True, then the data should be roughly in the interval [0,1]
            if False, then the data should be roughly in the interval [-1,1]
            (default: True)
        should_be_clipped : bool, optional
            flag to return a clipped output in range of [0,1]
            (default: False)
        eps : float, optional
            epsilon to prevent division by 0
            (default: 1e-20)


    Methods
    -------
    __call__(data)
        Applies the percentile based normalization
    """

    def __init__(self, mi, ma, should_be_zero_one_interval=True,
            should_be_clipped=False, eps=1e-20):

        assert not(((should_be_clipped==True) and
                (should_be_zero_one_interval==False))), "data shouldn't be \
        clipped in [0,1] interval, when data is normalized in a [-1,1] interval"

        self.mi = mi
        self.ma = ma

        self.should_be_zero_one_interval = should_be_zero_one_interval
        self.should_be_clipped = should_be_clipped
        self.eps = eps

    def __call__(self, data):
        """
        Applies the percentile based normalization

        Parameters
        ----------
        data : numpy.ndarray/torch.Tensor
            original image, which should be transformed

        Returns
        -------
        data : numpy.ndarray/torch.Tensor
            normalized data
        """

        if self.should_be_zero_one_interval:
            data = (data - self.mi) / (self.ma - self.mi + self.eps)
        else:
            data = 2 * ((data - self.mi) / (self.ma - self.mi + self.eps)) - 1

        if isinstance(data, np.ndarray):
            data = data.astype(np.float32, copy=False)
        else:
            data = data.float()

        if self.should_be_clipped:
            if isinstance(data, np.ndarray):
                data = np.clip(data, 0, 1)
            else:
                data = torch.clip(data, 0, 1)

        return data


### Flipping

class RandomFlip3D(object):
    """
        Flipping the data randomly left to right and up to top

    Attributes
    ----------

    Methods
    -------
    __call__(data)
        Applies the random-flipping of the data
    """

    def __call__(self, data):
        """
        Applies the random-flipping of the data

        Parameters
        ----------
        data : numpy.ndarray/torch.Tensor
            image-data that should be randomly flipped

        Returns
        -------
        data : numpy.ndarray/torch.Tensor
            flipped image-data. Batch, z or channel dimension stay unmodified.
        """

        if isinstance(data, torch.Tensor):
            if torch.rand(1) > 0.5:
                data = torch.fliplr(data)

            if torch.rand(1) > 0.5:
                data = torch.flipud(data)

        else:
            if np.random.rand() > 0.5:
                data = data[..., ::-1, :]

            if np.random.rand() > 0.5:
                data = data[..., ::-1, :, :]

        return data


class RandomCrop3D(object):
    """
        Crop the patches of specific size randomly

    Attributes
    ----------
        output_size : tuple, int
            shape tuple of the cropped patch (new_height, new_width)
            int-value of the squared cropped patch (new_height, new_height)

    Methods
    -------
    __call__(data)
        Applies the random-cropping of the data
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        """
        Applies the random-cropping of the data

        Parameters
        ----------
        data : numpy.ndarray/torch.Tensor
            image-data, from which a randomly patch is cropped

        Returns
        -------
        data : numpy.ndarray/torch.Tensor
            cropped patch with the wanted output_size. Batch, z or
            channel dimension stay unmodified.
        """

        new_h, new_w = self.output_size

        if isinstance(data, torch.Tensor):
            if len(data.size()) < 3:
                h, w = data.size()
            else:
                h, w = data.size()[-2:]
        else:
            if len(data.shape) < 3:
                h, w = data.shape
            else:
                h, w = data.shape[-3:-1]


        if isinstance(data, torch.Tensor):

            top = torch.randint(0, h - new_h, (1,)).item()
            left = torch.randint(0, w - new_w, (1,)).item()

            id_y = torch.arange(top, top + new_h, 1)[:, np.newaxis].long()
            id_x = torch.arange(left, left + new_w, 1).long()

            data = data[..., id_y, id_x]

        else:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
            id_x = np.arange(left, left + new_w, 1).astype(np.int32)

            if len(data.shape) < 3:
                data = data[id_y, id_x]
            else:
                data = data[..., id_y, id_x, :]

        return data


## N2V augmenation

# Function to generate structured noise mask (median blind spotting)
class N2V_mask_generator(object):

    def __init__(self, shape, nch=1, perc_pixel=20, n2v_neighborhood_radius=5, blindspot_strategy="median", structN2Vmask=None):
        self.shape = shape
        # Exclude channel dimension
        self.dims = len(self.shape)
        self.n_chan = nch
        self.local_sub_patch_radius = n2v_neighborhood_radius
        self.blindspot_strategy = blindspot_strategy
        self.structN2Vmask = structN2Vmask

        num_pix = int(np.product(self.shape) * self.n_chan/100.0 * perc_pixel)
        print("Number of blindspots:    ", num_pix)
        assert num_pix >= 1, "Number of blind-spot pixels is below one. At least {}% of pixels should be replaced.".format(100.0/(np.product(self.shape)*self.n_chan))

        if self.dims == 2:
            self.box_size = np.round(np.sqrt(100/perc_pixel)).astype(np.int16)
            self.get_stratified_coords = self.__get_stratified_coords2D__
            self.rand_float = self.__rand_float_coords2D__(self.box_size)
        elif self.dims == 3:
            self.box_size = np.round(np.sqrt(100 / perc_pixel)).astype(np.int16)
            self.get_stratified_coords = self.__get_stratified_coords3D__
            self.rand_float = self.__rand_float_coords3D__(self.box_size)
        else:
            raise Exception('Dimensionality not supported.')

    # Code needed for the indexing and finding void pixels
    # return x_val_structN2V, indexing_structN2V
    @staticmethod
    def __get_stratified_coords2D__(coord_gen, box_size, shape):
        box_count_y = int(np.ceil(shape[0] / box_size))
        box_count_x = int(np.ceil(shape[1] / box_size))
        x_coords = []
        y_coords = []
        for i in range(box_count_y):
            for j in range(box_count_x):
                y, x = next(coord_gen)
                y = int(i * box_size + y)
                x = int(j * box_size + x)
                if (y < shape[0] and x < shape[1]):
                    y_coords.append(y)
                    x_coords.append(x)
        return (y_coords, x_coords)

    @staticmethod
    def __get_stratified_coords3D__(coord_gen, box_size, shape):
        box_count_z = int(np.ceil(shape[0] / box_size))
        box_count_y = int(np.ceil(shape[1] / box_size))
        box_count_x = int(np.ceil(shape[2] / box_size))
        x_coords = []
        y_coords = []
        z_coords = []
        for i in range(box_count_z):
            for j in range(box_count_y):
                for k in range(box_count_x):
                    z, y, x = next(coord_gen)
                    z = int(i * box_size + z)
                    y = int(j * box_size + y)
                    x = int(k * box_size + x)
                    if (z < shape[0] and y < shape[1] and x < shape[2]):
                        z_coords.append(z)
                        y_coords.append(y)
                        x_coords.append(x)
        return (z_coords, y_coords, x_coords)

    @staticmethod
    def __rand_float_coords2D__(boxsize):
        while True:
            yield (np.random.rand() * boxsize, np.random.rand() * boxsize)

    @staticmethod
    def __rand_float_coords3D__(boxsize):
        while True:
            yield (np.random.rand() * boxsize, np.random.rand() * boxsize, np.random.rand() * boxsize)

    def __call__(self, data):

        # Use since we need a channel dimension
        label = data
        input = copy.deepcopy(label)
        # Use since we need a channel dimension
        mask = np.ones((label.shape), dtype=np.float32)

        for c in range(self.n_chan):
            coords = self.get_stratified_coords(self.rand_float, box_size=self.box_size,
                                                shape=self.shape)

            indexing = coords + (c,)
            indexing_mask = coords + (c,)
            # Picking strategy
            if self.blindspot_strategy == "random":
                # Pick randomly
                value_manipulation = self.pm_uniform_withCP()
            elif self.blindspot_strategy == "median":
                # Pick the median of the patch
                value_manipulation = self.pm_median()
            else:
                # Pick the mean of the patch
                value_manipulation = self.pm_mean()

            # Since the channel dimension is not taken into account, we need to reduce the self.dims
            input_val = value_manipulation(input[..., c], coords, self.dims)

            input[indexing] = input_val
            mask[indexing_mask] = 0

            if self.structN2Vmask is not None:
                self.apply_structN2Vmask(input[..., c], coords, self.structN2Vmask)

        return {'input': input, 'label': label, 'mask': mask}

    # Methods to determine the pixel value of the void pixel
    # The used fill in strategy (for more look at https://github.com/juglab/n2v/blob/master/n2v/utils/n2v_utils.py or https://github.com/juglab/n2v/blob/master/examples/2D/denoising2D_RGB/01_training.ipynb)
    # Different sample strategy (See: N2V2 - Fixing Noise2Void Checkerboard Artifacts with Modified Sampling Strategies and a Tweaked Network Architecture, taken from https://github.com/juglab/n2v/blob/master/n2v/utils/n2v_utils.py) 
    def mask_center(self, ndims=2):
        size = self.local_sub_patch_radius*2 + 1
        patch_wo_center = np.ones((size, ) * ndims)
        if ndims == 2:
            patch_wo_center[self.local_sub_patch_radius, self.local_sub_patch_radius] = 0
        elif ndims == 3:
            patch_wo_center[self.local_sub_patch_radius,
            self.local_sub_patch_radius, self.local_sub_patch_radius] = 0
        else:
            raise NotImplementedError()
        return ma.make_mask(patch_wo_center)
 
    # Picking random blind spot values from neighborhood
    def pm_uniform_withCP(self):
        def random_neighbor_withCP_uniform(patch, coords, dims):
            vals = []
            for coord in zip(*coords):
                sub_patch, _, _ = self.get_subpatch(patch, coord, self.local_sub_patch_radius)
                rand_coords = [np.random.randint(0, s) for s in sub_patch.shape[0:dims]]
                vals.append(sub_patch[tuple(rand_coords)])
            return vals
        return random_neighbor_withCP_uniform

    # Picking average value from neighborhood as blind spot value
    def pm_mean(self):
        def patch_mean(patch, coords, dims, structN2Vmask=None):
            patch_wo_center = self.mask_center(ndims=dims)
            vals = []
            for coord in zip(*coords):
                sub_patch, crop_neg, crop_pos = self.get_subpatch(patch, coord, self.local_sub_patch_radius)
                slices = [slice(-n, s-p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
                sub_patch_mask = (structN2Vmask or patch_wo_center)[tuple(slices)]
                vals.append(np.mean(sub_patch[sub_patch_mask]))
            return vals
        return patch_mean

    # Picking median value from neighborhood as blind spot value
    def pm_median(self):
        def patch_median(patch, coords, dims, structN2Vmask=None):
            patch_wo_center = self.mask_center(ndims=dims)
            vals = []
            for coord in zip(*coords):
                sub_patch, crop_neg, crop_pos = self.get_subpatch(patch, coord, self.local_sub_patch_radius)
                slices = [slice(-n, s-p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
                sub_patch_mask = (structN2Vmask or patch_wo_center)[tuple(slices)]
                vals.append(np.median(sub_patch[sub_patch_mask]))
            return vals

        return patch_median

    # Random crop
    def random_crop(self, data):
        h, w = data.shape[:2]
        new_h, new_w = self.shape[:2]

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        return data[id_y, id_x]


    def apply_structN2Vmask(self, patch, coords, mask):
        """
        each point in coords corresponds to the center of the mask.
        then for point in the mask with value=1 we assign a random value
        """
        coords = np.array(coords).astype(np.int)
        ndim = mask.ndim
        center = np.array(mask.shape)//2
        ## leave the center value alone
        mask[tuple(center.T)] = 0
        ## displacements from center
        dx = np.indices(mask.shape)[:,mask==1] - center[:,None]
        ## combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
        mix = (dx.T[...,None] + coords[None])
        mix = mix.transpose([1,0,2]).reshape([ndim,-1]).T
        ## stay within patch boundary
        mix = mix.clip(min=np.zeros(ndim),max=np.array(patch.shape)-1).astype(np.uint)
        ## replace neighbouring pixels with random values from flat dist (since the data is in range of 0,1, ideally)
        ## --> Think of a different strategy
        patch[tuple(mix.T)] = np.random.rand(mix.shape[0])*1.4 - np.random.rand(mix.shape[0])*0.1#*4 - 2

    # methods to determine the pixel value of the void pixel
    def get_subpatch(self, patch, coord, local_sub_patch_radius, crop_patch=True):
        crop_neg, crop_pos = 0, 0
        if crop_patch:
            start = np.array(coord) - local_sub_patch_radius
            end = start + local_sub_patch_radius * 2 + 1

            # compute offsets left/up ...
            crop_neg = np.minimum(start, 0)
            # and right/down
            crop_pos = np.maximum(0, end-patch.shape)

            # correct for offsets, patch size shrinks if crop_*!=0
            start -= crop_neg
            end -= crop_pos
        else:
            start = np.maximum(0, np.array(coord) - local_sub_patch_radius)
            end = start + local_sub_patch_radius * 2 + 1

            shift = np.minimum(0, patch.shape - end)

            start += shift
            end += shift

        slices = [slice(s, e) for s, e in zip(start, end)]

        # return crop vectors for deriving correct center pixel locations later
        return patch[tuple(slices)], crop_neg, crop_pos


class ToTensor3D(object):
    """
        Converting np.ndarray to torch.Tensor

    Attributes
    ----------

    Methods
    -------
    __call__(data)
        Applies the conversion of the data
    """

    def __call__(self, data):
        """
        Applies the conversion of the data

        Parameters
        ----------
        data : numpy.ndarray, data dict
            image-data that should be converted

        Returns
        -------
        data : torch.Tensor
            converted image-data. Batch dimension stay unmodified.
            Swap channel axis because   numpy image: Z x H x W x C
                                        torch image: C x Z x H x W

        """
        assert isinstance(data, np.ndarray) or isinstance(data, dict) , "data needs to be numpy.ndarray"

        # If N2V approach
        if isinstance(data, dict):
            
            input, label, mask = data['input'], data['label'], data['mask']
            
            # No Batch dimension
            if len(input.shape) == 4:
                input = np.moveaxis(input, [0,1,2,3], [1,2,3,0])
                label = np.moveaxis(label, [0,1,2,3], [1,2,3,0])
                mask = np.moveaxis(mask, [0,1,2,3], [1,2,3,0])
            else:
                input = np.moveaxis(input, [0,1,2,3,4], [0,2,3,4,1])
                label = np.moveaxis(label, [0,1,2,3,4], [0,2,3,4,1])
                mask = np.moveaxis(mask, [0,1,2,3,4], [0,2,3,4,1])

            data = {'input': torch.from_numpy(input), 'label': torch.from_numpy(label), 'mask': torch.from_numpy(mask)}
        else:
            # No Batch dimension
            if len(data.shape) == 4:
                data = np.moveaxis(data, [0,1,2,3], [1,2,3,0])
            else:
                data = np.moveaxis(data, [0,1,2,3,4], [0,2,3,4,1])

            data = torch.from_numpy(data)

        return data


class ToNumpy3D(object):
    """
        Converting torch.Tensor to np.ndarray

    Attributes
    ----------

    Methods
    -------
    __call__(data)
        Applies the conversion of the data
    """

    def __call__(self, data):
        """
        Applies the conversion of the data

        Parameters
        ----------
        data : torch.Tensor
            image-data that should be converted

        Returns
        -------
        data : numpy.ndarray
            converted image-data. Batch dimension stay unmodified.
            Swap channel axis because   torch image: C x Z x H x W
                                        numpy image: Z x H x W x C

        """

        assert isinstance(data,
                torch.Tensor), "data needs to be torch.Tensor"

        if len(data.size()) == 4:
            data = np.moveaxis(data.to('cpu').detach().numpy(), [0,1,2,3], [3,0,1,2])
        else:
            data = np.moveaxis(data.to('cpu').detach().numpy(), [0,1,2,3,4], [0,4,1,2,3])

        return data


## Denormalization

class PercentileDenormalize3D(object):
    """
        Percentile based denormalization

    Attributes
    ----------
        mi : float
            lower limit used for the normalisation
        ma : float
            top limit used for the normalisation
        is_zero_one_interval : bool, optional
            flag to show if the data is in a certain interval
            if True, then the data is roughly in the interval [0,1]
            if False, then the data is roughly in the interval [-1,1]
            (default: True)
        should_be_clipped : bool, optional
            flag to return a clipped output in range of [0,1]
            (default: False)
        eps : float, optional
            epsilon to prevent division by 0
            (default: 1e-20)


    Methods
    -------
    __call__(data)
        Applies the percentile based normalization
    """

    def __init__(self, mi, ma, is_zero_one_interval=True,
            should_be_clipped=False, eps=1e-20):

        self.mi = mi
        self.ma = ma

        self.eps = eps
        self.is_zero_one_interval = is_zero_one_interval
        self.should_be_clipped = should_be_clipped

    def __call__(self, data):
        """
        Applies the percentile based denormalization

        Parameters
        ----------
        data : numpy.ndarray/torch.Tensor
            normlized image, which should be back-transformed

        Returns
        -------
        data : numpy.ndarray/torch.Tensor
            original data
        """

        if self.is_zero_one_interval:
            data = data  * (self.ma - self.mi + self.eps) + self.mi
        else:
            data = 0.5 * (data + 1) * (self.ma - self.mi + self.eps) + self.mi

        if isinstance(data, np.ndarray):
            data = data.astype(np.float32, copy=False)
        else:
            data = data.float()

        if self.should_be_clipped:
            if isinstance(data, np.ndarray):
                data = np.clip(data, 0, 1)
            else:
                data = torch.clip(data, 0, 1)

        return data


class ZCrop3D(object):
    """
        Crop the first and last n-th time frames of the 3D data

    Attributes
    ----------
        z_cropwidth : tuple, int
            crop width of the front and tail of the 3D data (crop_width_front, crop_width_tail)
            int-value of the crop_width of the 3D data (crop_width, crop_width)

    Methods
    -------
    __call__(data)
        Applies the cropping of the 3D data
    """

    def __init__(self, z_cropwidth):
        assert isinstance(z_cropwidth, (int, tuple))
        if isinstance(z_cropwidth, int):
            self.z_cropwidth = (z_cropwidth, z_cropwidth)
        else:
            assert len(z_cropwidth) == 2
            self.z_cropwidth = z_cropwidth

    def __call__(self, data):
        """
        Applies the cropping of the 3D data

        Parameters
        ----------
        data : numpy.ndarray/torch.Tensor
            image-stack-data, from which front and tail is cropped

        Returns
        -------
        data : numpy.ndarray/torch.Tensor
            cropped image_stack with the wanted croppwidth. Batch, x, y or
            channel dimension stay unmodified.
        """

        crop_front, crop_tail = self.z_cropwidth

        if isinstance(data, torch.Tensor):

            data = data[...,crop_front:-crop_tail, :, :]

        else:

            if len(data.shape) < 4:
                data = data[crop_front:-crop_tail]
            else:
                data = data[..., crop_front:-crop_tail, :,:,:]

        return data