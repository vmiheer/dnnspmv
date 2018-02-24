#!/usr/env python

"""
This is a wrapper function for the sampling
"""

import sys, os
import numpy as np
from cffi import FFI


DEBUG = False
LIBPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                       'libmat_sample.so')

def dbg(data):
    """[summary]

    [description]

    Arguments:
        data {[type]} -- [description]
    """

    if DEBUG:
        print(data)


def meta_to_dict(meta):
    """[summary] convert meta data to python dict

    [description]

    Arguments:
        meta {[type]} -- [description]
    """

    return {'id': None, 'M': meta.M, 'N': meta.N, 'nz': meta.nz, 'scalef': meta.scalef}


class DlSample(object):
    """[summary]

    [description]
    """

    def __init__(self):
        self.ffi = FFI()
        self.lib = self.ffi.dlopen(LIBPATH)
        # dir(lib)
        self.ffi.cdef("""
        typedef struct {
          int M;
          int N;
          int nz;
          double scalef;
        } SpMeta;
        int mat_sample(char* mtxfile, int output_resolution, int** Img, SpMeta *meta);
        """)

    def sample(self, filename, output_resolution):
        """ return data of one matrix
        """
        dbg(filename)
        meta = self.ffi.new('SpMeta *')
        # !IMPORTANT: int32, not int
        img = np.zeros((output_resolution, output_resolution), dtype='int32')

        # get pointers to the numpy underlying data structure
        img_row_ptrs = self.ffi.new('int* [%d]' % (img.shape[0]))
        img_store_ptr = self.ffi.cast('int *', img.ctypes.data)

        for i in range(img.shape[0]):
            img_row_ptrs[i] = img_store_ptr + i * img.shape[1]

        # !filename must be bytes (b''), use some_str.encode() for string
        self.lib.mat_sample(filename, output_resolution, img_row_ptrs, meta)

        return meta_to_dict(meta), img

    def batch(self, filelist, output_resolution):
        """ return data of a batch of matrices
        """
        with open(filelist) as filenames:
            files = [filename.strip() for filename in filenames.readlines()]

        batch_size = len(files)
        dbg(batch_size)
        meta_batch = []
        img_batch = np.zeros((batch_size, output_resolution, output_resolution), dtype='int32')

        for findex, filename in enumerate(files):
            dbg(findex)
            dbg(filename)
            matid = int(filename.split('/')[-1].split('.')[0])
            dbg(matid)
            meta, img = self.sample(filename.encode(), output_resolution)
            meta['id'] = matid

            meta_batch.append(meta)
            dbg(meta_batch[-1])
            dbg(img)
            img_batch[findex, :, :] = img

        return meta_batch, img_batch


if __name__ == '__main__':
    """[summary]

    [description]
    """
    if len(sys.argv) < 3:
        print("Usage: {} <matrix.list> <resolution>".format(sys.argv[0]))
        exit(1)

    MATRIXLIST = sys.argv[1] # '../test/Origin.list'
    RES = int(sys.argv[2]) # 64

    if os.path.isfile(MATRIXLIST):
        try:
            sampler = DlSample()
            metas, imgs = sampler.batch(MATRIXLIST, RES)
            dbg(imgs.ndim)
            np.savez('data{}.npz'.format(RES), metas=metas, imgs=imgs)
        except:
            print("Error")
            exit(1)
