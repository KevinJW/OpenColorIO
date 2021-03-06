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

#include <cstring>
#include <sstream>

#include "fileformats/ctf/CTFReaderUtils.h"
#include "Platform.h"

OCIO_NAMESPACE_ENTER
{

namespace
{
static constexpr const char * INTERPOLATION_1D_LINEAR = "linear";
static constexpr const char * INTERPOLATION_1D_CUBIC = "cubic";
static constexpr const char * INTERPOLATION_DEFAULT = "default";

static constexpr const char * INTERPOLATION_3D_LINEAR = "trilinear";
static constexpr const char * INTERPOLATION_3D_TETRAHEDRAL = "tetrahedral";

}

Interpolation GetInterpolation1D(const char * str)
{
    if (str && *str)
    {
        if (0 == Platform::Strcasecmp(str, INTERPOLATION_1D_LINEAR))
        {
            return INTERP_LINEAR;
        }
        else if (0 == Platform::Strcasecmp(str, INTERPOLATION_1D_CUBIC))
        {
            return INTERP_CUBIC;
        }
        else if (0 == Platform::Strcasecmp(str, INTERPOLATION_DEFAULT))
        {
            return INTERP_DEFAULT;
        }

        std::ostringstream oss;
        oss << "1D LUT interpolation not recongnized: '" << str << "'.";
        throw Exception(oss.str().c_str());
    }

    throw Exception("1D LUT missing interpolation value.");
}

const char * GetInterpolation1DName(Interpolation interp)
{
    switch (interp)
    {
    case INTERP_LINEAR:
        return INTERPOLATION_1D_LINEAR;
    case INTERP_CUBIC:
        return INTERPOLATION_1D_CUBIC;
    case INTERP_DEFAULT:
    default:
        return INTERPOLATION_DEFAULT;
    break;
    };
    return INTERPOLATION_DEFAULT;
}

Interpolation GetInterpolation3D(const char * str)
{
    if (str && *str)
    {
        if (0 == Platform::Strcasecmp(str, INTERPOLATION_3D_LINEAR))
        {
            return INTERP_LINEAR;
        }
        else if (0 == Platform::Strcasecmp(str, INTERPOLATION_3D_TETRAHEDRAL))
        {
            return INTERP_TETRAHEDRAL;
        }
        else if (0 == Platform::Strcasecmp(str, INTERPOLATION_DEFAULT))
        {
            return INTERP_DEFAULT;
        }

        std::ostringstream oss;
        oss << "3D LUT interpolation not recongnized: '" << str << "'.";
        throw Exception(oss.str().c_str());
    }

    throw Exception("3D LUT missing interpolation value.");
}

const char * GetInterpolation3DName(Interpolation interp)
{
    switch (interp)
    {
    case INTERP_LINEAR:
        return INTERPOLATION_3D_LINEAR;
    case INTERP_CUBIC:
        return INTERPOLATION_3D_TETRAHEDRAL;
    case INTERP_DEFAULT:
    default:
        return INTERPOLATION_DEFAULT;
        break;
    };
    return INTERPOLATION_DEFAULT;
}

}
OCIO_NAMESPACE_EXIT
