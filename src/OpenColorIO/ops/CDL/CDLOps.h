/*
Copyright (c) 2018 Autodesk Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef INCLUDED_OCIO_CDL_H
#define INCLUDED_OCIO_CDL_H

#include <vector>

#include <OpenColorIO/OpenColorIO.h>

#include "Op.h"
#include "ops/CDL/CDLOpData.h"


OCIO_NAMESPACE_ENTER
{

// Create a CDL op from its parameters and append it to ops.
void CreateCDLOp(OpRcPtrVec & ops,
                 const FormatMetadataImpl & info,
                 CDLOpData::Style style,
                 const double * slope3,
                 const double * offset3,
                 const double * power3,
                 double saturation,
                 TransformDirection direction);

// Create a CDL op from the CDL OpData and append it to ops. FormatMetadata is preserved.
void CreateCDLOp(OpRcPtrVec & ops, 
                 CDLOpDataRcPtr & cdlData,
                 TransformDirection direction);

// Create a copy of the CDL transform in the op and append it to the GroupTransform.
void CreateCDLTransform(GroupTransformRcPtr & group, ConstOpRcPtr & op);

}
OCIO_NAMESPACE_EXIT

#endif
