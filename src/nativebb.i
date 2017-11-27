/* 
 * This file is part of the 3D-MANC package for brainbow image segmentation
 *   (https://github.com/lucasgroup/3D-MANC)
 *
 * Copyright (c) 2017 Egor Zindy <egor.zindy@manchester.ac.uk>
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * vim: set ts=4 sts=4 sw=4 expandtab smartindent:
 */

%module nativebb
%{
#include <errno.h>
#include <stdint.h>
#include "nativebb.h"

#define SWIG_FILE_WITH_INIT
%}

%include "numpy_mod.i"
%include "exception.i"

%init %{
    import_array();
%}

%exception
{
    errno = 0;
    $action

    if (errno != 0)
    {
        switch(errno)
        {
            case E2BIG:
                PyErr_Format(PyExc_ValueError, "All three images must have the same dimensions");
                break;
            case ENOMEM:
                PyErr_Format(PyExc_MemoryError, "Not enough memory");
                break;
            case EPERM:
                PyErr_Format(PyExc_IndexError, "Unknown index value");
                break;
            default:
                PyErr_Format(PyExc_Exception, "Unknown exception");
        }
        SWIG_fail;
    }
}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *arr_in, int n_in, int w_in)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *poly, int n_poly, int w_poly)}
%apply (int** ARGOUTVIEWM_ARRAY1, int* DIM1) {(int **arr_out, int *n_out)}

%apply (uint16_t* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(uint16_t *ArrayR, int ZdimR, int YdimR, int XdimR)}
%apply (uint16_t* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(uint16_t *ArrayG, int ZdimG, int YdimG, int XdimG)}
%apply (uint16_t* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(uint16_t *ArrayB, int ZdimB, int YdimB, int XdimB)}
%apply (uint16_t** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(uint16_t **ArrayOut, int *YdimOut, int *XdimOut)}
%apply (uint64_t** ARGOUTVIEWM_ARRAY1, int* DIM1) {(uint64_t **ArrayOut, int *NdimOut)}

%rename (dedupcol) dedupcol_safe;
%rename (dedupcol_indexes) dedupcol_indexes_safe;

%inline %{

void dedupcol_safe(
        uint16_t *ArrayR, int ZdimR, int YdimR, int XdimR,
        uint16_t *ArrayG, int ZdimG, int YdimG, int XdimG,
        uint16_t *ArrayB, int ZdimB, int YdimB, int XdimB,
        int input_bit, int output_bit,
        uint16_t **ArrayOut, int *YdimOut, int *XdimOut)
{

    int _ZdimR = (ZdimR == -1)?1:ZdimR;
    int _YdimR = (YdimR == -1)?1:YdimR;
    int _ZdimG = (ZdimG == -1)?1:ZdimG;
    int _YdimG = (YdimG == -1)?1:YdimG;
    int _ZdimB = (ZdimB == -1)?1:ZdimB;
    int _YdimB = (YdimB == -1)?1:YdimB;

    dedupcol(
        ArrayR, _ZdimR, _YdimR, XdimR,
        ArrayG, _ZdimG, _YdimG, XdimG,
        ArrayB, _ZdimB, _YdimB, XdimB,
        input_bit, output_bit,
        ArrayOut, YdimOut, XdimOut);
}

void dedupcol_indexes_safe(
        uint16_t *ArrayR, int ZdimR, int YdimR, int XdimR,
        uint16_t *ArrayG, int ZdimG, int YdimG, int XdimG,
        uint16_t *ArrayB, int ZdimB, int YdimB, int XdimB,
        int input_bit, int output_bit,
        uint64_t **ArrayOut, int *NdimOut)
{

    int _ZdimR = (ZdimR == -1)?1:ZdimR;
    int _YdimR = (YdimR == -1)?1:YdimR;
    int _ZdimG = (ZdimG == -1)?1:ZdimG;
    int _YdimG = (YdimG == -1)?1:YdimG;
    int _ZdimB = (ZdimB == -1)?1:ZdimB;
    int _YdimB = (YdimB == -1)?1:YdimB;

    dedupcol_indexes(
        ArrayR, _ZdimR, _YdimR, XdimR,
        ArrayG, _ZdimG, _YdimG, XdimG,
        ArrayB, _ZdimB, _YdimB, XdimB,
        input_bit, output_bit,
        ArrayOut, NdimOut);
}

%}

%ignore dedupcol;
%ignore dedupcol_indexes;
%include "nativebb.h"

