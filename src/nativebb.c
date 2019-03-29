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

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <omp.h>

#include "nativebb.h"

//Returns an array of indexes of the 2-D points inside a 2-D polyon
void isinside(float *arr_in, int n_in, int w_in, float *poly, int n_poly, int w_poly, int **arr_out, int *n_out) {

    int *arr = NULL;
    int i,j,c,ipt,ipol;
    float x_pt;
    float y_pt;
    float xi,yi,xj,yj;

    if (w_in != 2 || w_poly !=2) {
        errno = EPERM;
        goto end;
    }

    //allocating some memory for the output array
    if (*arr_out == NULL)
        arr = (int *)malloc(n_in*w_in*sizeof(int));
    else
        arr = *arr_out;

    #pragma omp parallel for        \
        default(shared) private(ipt,c,x_pt,y_pt,j,ipol,i,xj,yj,xi,yi)

    for (ipt=0; ipt<n_in; ipt++) {
        c=0;
        x_pt = arr_in[ipt*2];
        y_pt = arr_in[ipt*2+1];
        j=2*(n_poly-1);
        for (ipol=0; ipol<n_poly; ipol++)
        {
            i = 2*ipol;
            xj=(float)(poly[j]);
            yj=(float)(poly[j+1]);
            xi=(float)(poly[i]);
            yi=(float)(poly[i+1]);

            if ((((yi<=y_pt) && (y_pt<yj)) || ((yj<=y_pt) && (y_pt<yi))) && \
                    (x_pt < (xj - xi) * (y_pt - yi) / (yj - yi) + xi))
                c = 1-c;
            j=i;
        }
        arr[ipt] = c;
    }

end:
    *arr_out = arr;
    *n_out = n_in;
}

//takes three arrays (RGB) and a max value, returns a Nx3 list of unique RGB values
void dedupcol(
        uint16_t *ArrayR, int ZdimR, int YdimR, int XdimR,
        uint16_t *ArrayG, int ZdimG, int YdimG, int XdimG,
        uint16_t *ArrayB, int ZdimB, int YdimB, int XdimB,
        int input_bit, int output_bit, int threshold,
        uint16_t **ArrayOut, int *YdimOut, int *XdimOut)
{
    uint64_t i,j,n=0;

    //output array
    uint16_t* decomp = NULL;

    int decimation = input_bit - output_bit;
    decimation = (decimation > 0)?decimation:0;

    uint64_t max_out = pow(2,output_bit)-1;
    uint64_t signature, temp_r, temp_g, temp_b, max_signature=0;

    //This is the truth array that holds all three RGB values
    uint8_t* truth = NULL;

    if ( ZdimR != ZdimG || ZdimR != ZdimB || YdimR != YdimG || YdimR != YdimB || XdimR != XdimG || XdimR != XdimB) {
        errno = E2BIG; goto end;
    }
    
    //The truth table needs to encompass all the values from 0 to 2**outputbits in all three dimensions.
    truth = (uint8_t *)calloc(pow(2,(output_bit*3)), sizeof(uint8_t));
    if (truth == NULL) { errno = ENOMEM; goto end; }

    //The strategy here is to fill truth array if we have a hit. Simultaneously, we could have multiple threads hitting the same
    //truth index. So don't count the number of hits as part of this loop, we can do that afterwards
    #pragma omp parallel for        \
        default(shared) private(temp_r, temp_g, temp_b, signature)

    for (i=0; i<ZdimR*YdimR*XdimR; i++)
    {
        temp_r = (uint64_t)ArrayR[i] >> decimation;
        if (temp_r > max_out) temp_r = max_out;

        temp_g = (uint64_t)ArrayG[i] >> decimation;
        if (temp_g > max_out) temp_g = max_out;

        temp_b = (uint64_t)ArrayB[i] >> decimation;
        if (temp_b > max_out) temp_b = max_out;

        //construct the rgb signature
        signature = temp_r | (temp_g << output_bit) | (temp_b << (output_bit*2));

        /*
        if (truth[signature] == 1)
            continue;
        */

        if (signature > max_signature) {
            max_signature = signature;
        }

        //printf("signature=%d max_sig=%d\n",signature,max_signature);
        truth[signature] += 1;
    }

    //we need to count the total number of hits, this is to determine the size of the output array (nx3).
    n = 0;
    #pragma omp parallel for        \
        default(shared) reduction( + : n )

    for (i=0; i<=max_signature; i++) {
        if (truth[i] > threshold)
            n++;
    }

    //output is 16 bit (up to)
    //could definitely combine these both into a single realloc. This just reminds me what is going on.
    if (*ArrayOut == NULL) {
        decomp = (uint16_t *)malloc(3*n*sizeof(uint16_t));
    } else {
        decomp = (uint16_t *)realloc(*ArrayOut, 3*n*sizeof(uint16_t));
    }
    if (decomp == NULL) { errno = ENOMEM; goto end; }

    //use this as a mask of width output_bit to unpack the index back into 3 r,g,b values
    signature = pow(2,output_bit) - 1;

    j = 0;
    for (i=0; i<=max_signature; i++)
    {
        if (truth[i] > threshold)
        {
            //printf("i=%d j=%d truth[i]=%d\n",i,j,truth[i]);
            temp_r = i & signature;
            temp_g = (i >> output_bit) & signature;
            temp_b = (i >> (output_bit*2)) & signature;

            decomp[j++] = (uint16_t)temp_r;
            decomp[j++] = (uint16_t)temp_g;
            decomp[j++] = (uint16_t)temp_b;
            truth[i] = 0;
        }
    }

end:
    if (truth != NULL) free(truth);

    *ArrayOut = decomp;
    *YdimOut = n;
    *XdimOut = 3;
}

//This re
void dedupcol_indexes(
        uint16_t *ArrayR, int ZdimR, int YdimR, int XdimR,
        uint16_t *ArrayG, int ZdimG, int YdimG, int XdimG,
        uint16_t *ArrayB, int ZdimB, int YdimB, int XdimB,
        int input_bit, int output_bit, int threshold,
        uint64_t **ArrayOut, int *NdimOut)
{
    uint64_t i,j,n=0;

    //output array
    uint64_t* decomp = NULL;

    int decimation = input_bit - output_bit;
    decimation = (decimation > 0)?decimation:0;

    uint64_t max_out = pow(2,output_bit)-1;
    uint64_t signature, temp_r, temp_g, temp_b, max_signature=0;

    //This is the truth array that holds all three RGB values
    uint8_t* truth = NULL;
    uint64_t* truth_index = NULL;

    if ( ZdimR != ZdimG || ZdimR != ZdimB || YdimR != YdimG || YdimR != YdimB || XdimR != XdimG || XdimR != XdimB) {
        errno = E2BIG; goto end;
    }
    
    //The truth table needs to encompass all the values from 0 to 2**outputbits in all three dimensions.
    truth = (uint8_t *)calloc(pow(2,(output_bit*3)), sizeof(uint8_t));
    truth_index = (uint64_t *)malloc(pow(2,(output_bit*3))*sizeof(uint64_t));
    if (truth == NULL || truth_index == NULL) { errno = ENOMEM; goto end; }

    //The strategy here is to fill truth array if we have a hit. Simultaneously, we could have multiple threads hitting the same
    //truth index. So don't count the number of hits as part of this loop, we can do that afterwards
    #pragma omp parallel for        \
        default(shared) private(temp_r, temp_g, temp_b, signature)

    for (i=0; i<ZdimR*YdimR*XdimR; i++)
    {
        temp_r = (uint64_t)ArrayR[i] >> decimation;
        if (temp_r > max_out) temp_r = max_out;

        temp_g = (uint64_t)ArrayG[i] >> decimation;
        if (temp_g > max_out) temp_g = max_out;

        temp_b = (uint64_t)ArrayB[i] >> decimation;
        if (temp_b > max_out) temp_b = max_out;

        //construct the rgb signature
        signature = temp_r | (temp_g << output_bit) | (temp_b << (output_bit*2));

        /*
        if (truth[signature] == 1)
            continue;
        */

        if (signature > max_signature) {
            max_signature = signature;
        }

        truth[signature] += 1;
        truth_index[signature] = i;
        //printf("signature=%d max_sig=%d i=%d truth_index[signature]=%d\n",signature,max_signature,i, truth_index[signature]);
    }

    //we need to count the total number of hits, this is to determine the size of the output array (nx3).
    n = 0;
    #pragma omp parallel for        \
        default(shared) reduction( + : n )

    for (i=0; i<=max_signature; i++) {
        if (truth[i] > threshold)
            n++;
    }

    //output is 16 bit (up to)
    //could definitely combine these both into a single realloc. This just reminds me what is going on.
    if (*ArrayOut == NULL) {
        decomp = (uint64_t *)malloc(n*sizeof(uint64_t));
    } else {
        decomp = (uint64_t *)realloc(*ArrayOut, n*sizeof(uint64_t));
    }
    if (decomp == NULL) { errno = ENOMEM; goto end; }

    j = 0;
    for (i=0; i<=max_signature; i++)
    {
        if (truth[i] > threshold)
        {
            //printf("i=%d j=%d truth[i]=%d truth_index[i]=%d\n",i,j,truth[i],truth_index[i]);
            decomp[j] = truth_index[i];
            truth[i] = 0;
            j++;
        }

    }

end:
    if (truth != NULL) free(truth);
    if (truth_index != NULL) free(truth_index);

    *ArrayOut = decomp;
    *NdimOut = n;
}

