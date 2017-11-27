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

void isinside(
        float *arr_in, int n_in, int w_in,
        float *poly, int n_poly, int w_poly,
        int **arr_out, int *n_out);

void dedupcol(
        uint16_t *ArrayR, int ZdimR, int YdimR, int XdimR,
        uint16_t *ArrayG, int ZdimG, int YdimG, int XdimG,
        uint16_t *ArrayB, int ZdimB, int YdimB, int XdimB,
        int input_bit, int output_bit,
        uint16_t **ArrayOut, int *YdimOut, int *XdimOut);

void dedupcol_indexes(
        uint16_t *ArrayR, int ZdimR, int YdimR, int XdimR,
        uint16_t *ArrayG, int ZdimG, int YdimG, int XdimG,
        uint16_t *ArrayB, int ZdimB, int YdimB, int XdimB,
        int input_bit, int output_bit,
        uint64_t **ArrayOut, int *NdimOut);
