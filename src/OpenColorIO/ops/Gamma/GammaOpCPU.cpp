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

#include <algorithm>
#include <cmath>

#include <OpenColorIO/OpenColorIO.h>

#include "BitDepthUtils.h"
#include "ops/Gamma/GammaOpCPU.h"
#include "ops/Gamma/GammaOpUtils.h"

#include "SSE.h"


OCIO_NAMESPACE_ENTER
{

// Note: The parameters are validated when the op is created so that the
// math below does not require checks for divide by 0, etc.


// Base class for the Gamma (i.e. basic style) operation renderers.
class GammaBasicOpCPU : public OpCPU
{
public:
    GammaBasicOpCPU() = delete;
    GammaBasicOpCPU(const GammaBasicOpCPU &) = delete;
    explicit GammaBasicOpCPU(ConstGammaOpDataRcPtr & gamma);

    void apply(const void * inImg, void * outImg, long numPixels) const override;

protected:
    void update(ConstGammaOpDataRcPtr & gamma);

private:
    float m_inScale;
    float m_outScale;
    float m_redGamma;
    float m_grnGamma;
    float m_bluGamma;
    float m_alpGamma;
};

class GammaMoncurveOpCPU : public OpCPU
{
protected:
    explicit GammaMoncurveOpCPU(ConstGammaOpDataRcPtr &) : OpCPU() {}

protected:
    RendererParams m_red;
    RendererParams m_green;
    RendererParams m_blue;
    RendererParams m_alpha;
};

class GammaMoncurveOpCPUFwd : public GammaMoncurveOpCPU
{
public:
    explicit GammaMoncurveOpCPUFwd(ConstGammaOpDataRcPtr & gamma);

    void apply(const void * inImg, void * outImg, long numPixels) const override;

protected:
    void update(ConstGammaOpDataRcPtr & gamma);

private:
    float m_outScale;
};

class GammaMoncurveOpCPURev : public GammaMoncurveOpCPU
{
public:
    explicit GammaMoncurveOpCPURev(ConstGammaOpDataRcPtr & gamma);

    void apply(const void * inImg, void * outImg, long numPixels) const override;

protected:
    void update(ConstGammaOpDataRcPtr & gamma);

private:
    float m_inScale;
};


ConstOpCPURcPtr GetGammaRenderer(ConstGammaOpDataRcPtr & gamma)
{
    switch(gamma->getStyle())
    {
        case GammaOpData::MONCURVE_FWD:
        {
            return std::make_shared<GammaMoncurveOpCPUFwd>(gamma);
            break;
        }

        case GammaOpData::MONCURVE_REV:
        {
            return std::make_shared<GammaMoncurveOpCPURev>(gamma);
            break;
        }

        case GammaOpData::BASIC_FWD:
        case GammaOpData::BASIC_REV:
        {
            return std::make_shared<GammaBasicOpCPU>(gamma);
            break;
        }
    }

    throw Exception("Unsupported Gamma style");
}




GammaBasicOpCPU::GammaBasicOpCPU(ConstGammaOpDataRcPtr & gamma)
    :   OpCPU()
    ,   m_inScale(1.0f)
    ,   m_outScale(1.0f)
    ,   m_redGamma(0.0f)
    ,   m_grnGamma(0.0f)
    ,   m_bluGamma(0.0f)
    ,   m_alpGamma(0.0f)
{
    update(gamma);
}

void GammaBasicOpCPU::update(ConstGammaOpDataRcPtr & gamma)
{
    // The gamma calculations are done in normalized space.
    // Compute the scale factors for integer in/out depths.
    m_inScale = (float)(
        1. / GetBitDepthMaxValue(gamma->getInputBitDepth()));

    m_outScale = (float)(
        GetBitDepthMaxValue(gamma->getOutputBitDepth()));

    // Calculate the actual power used in the function.
    m_redGamma = (float)(
        gamma->getStyle() == GammaOpData::BASIC_FWD
        ? gamma->getRedParams()[0]
        : 1. / gamma->getRedParams()[0]);

    m_grnGamma = (float)(
        gamma->getStyle() == GammaOpData::BASIC_FWD 
        ? gamma->getGreenParams()[0] 
        : 1. / gamma->getGreenParams()[0]);

    m_bluGamma = (float)(
        gamma->getStyle() == GammaOpData::BASIC_FWD 
        ? gamma->getBlueParams()[0] 
        : 1. / gamma->getBlueParams()[0]);

    m_alpGamma = (float)(
        gamma->getStyle() == GammaOpData::BASIC_FWD 
        ? gamma->getAlphaParams()[0]
        : 1. / gamma->getAlphaParams()[0]);
}

void GammaBasicOpCPU::apply(const void * inImg, void * outImg, long numPixels) const
{
    const float * in = (const float *)inImg;
    float * out = (float *)outImg;

#ifdef USE_SSE
    const __m128 gamma = _mm_set_ps(m_alpGamma, m_bluGamma, m_grnGamma, m_redGamma);
      
    const __m128 inScale  = _mm_set1_ps(m_inScale);
    const __m128 outScale = _mm_set1_ps(m_outScale);

    for(long idx=0; idx<numPixels; ++idx)
    {
        __m128 pixel = _mm_set_ps(in[3], in[2], in[1], in[0]);

        pixel = _mm_mul_ps(pixel, inScale);

        pixel = ssePower(pixel, gamma);

        pixel = _mm_mul_ps(pixel, outScale);

        _mm_storeu_ps(out, pixel);

        in  += 4;
        out += 4;
    }
#else
    for(long idx=0; idx<numPixels; ++idx)
    {
        const float pixel[4] = { std::max(0.0f, in[0]), 
                                 std::max(0.0f, in[1]), 
                                 std::max(0.0f, in[2]),
                                 std::max(0.0f, in[3]) };

        out[0] = powf(pixel[0] * m_inScale, m_redGamma) * m_outScale;
        out[1] = powf(pixel[1] * m_inScale, m_grnGamma) * m_outScale;
        out[2] = powf(pixel[2] * m_inScale, m_bluGamma) * m_outScale;
        out[3] = powf(pixel[3] * m_inScale, m_alpGamma) * m_outScale;

        in  += 4;
        out += 4;
    }
#endif
}

GammaMoncurveOpCPUFwd::GammaMoncurveOpCPUFwd(ConstGammaOpDataRcPtr & gamma)
    :   GammaMoncurveOpCPU(gamma)
    ,   m_outScale(0.0f)
{
    update(gamma);
}

void GammaMoncurveOpCPUFwd::update(ConstGammaOpDataRcPtr & gamma)
{
    // NB: The power function is applied in normalized space
    // but we fold the in/out depth conversion into the other scaling
    // to minimize the number of multiplies.

    const BitDepth inBitDepth  = gamma->getInputBitDepth();
    const BitDepth outBitDepth = gamma->getOutputBitDepth();

    ComputeParamsFwd(gamma->getRedParams(),   inBitDepth, outBitDepth, m_red);
    ComputeParamsFwd(gamma->getGreenParams(), inBitDepth, outBitDepth, m_green);
    ComputeParamsFwd(gamma->getBlueParams(),  inBitDepth, outBitDepth, m_blue);
    ComputeParamsFwd(gamma->getAlphaParams(), inBitDepth, outBitDepth, m_alpha);

    m_outScale = (float)GetBitDepthMaxValue(outBitDepth);
}

void GammaMoncurveOpCPUFwd::apply(const void * inImg, void * outImg, long numPixels) const
{
    const float * in = (const float *)inImg;
    float * out = (float *)outImg;

#ifdef USE_SSE
    const __m128 scale
      = _mm_set_ps(m_alpha.scale, m_blue.scale,
                   m_green.scale, m_red.scale);

    const __m128 offset
      = _mm_set_ps(m_alpha.offset, m_blue.offset,
                   m_green.offset, m_red.offset);

    const __m128 gamma
      = _mm_set_ps(m_alpha.gamma, m_blue.gamma,
                   m_green.gamma, m_red.gamma);

    const __m128 breakPnt
      = _mm_set_ps(m_alpha.breakPnt, m_blue.breakPnt,
                   m_green.breakPnt, m_red.breakPnt);

    const __m128 slope
      = _mm_set_ps(m_alpha.slope, m_blue.slope,
                   m_green.slope, m_red.slope);

    const __m128 outScale = _mm_set1_ps(m_outScale);

    for(long idx=0; idx<numPixels; ++idx)
    {
        __m128 pixel = _mm_set_ps(in[3], in[2], in[1], in[0]);

        __m128 data = _mm_add_ps(_mm_mul_ps(pixel, scale), offset);

        data = ssePower(data, gamma);

        data = _mm_mul_ps(data, outScale);

        __m128 flag = _mm_cmpgt_ps( pixel, breakPnt);

        data = _mm_or_ps(_mm_and_ps(flag, data),
                         _mm_andnot_ps(flag, _mm_mul_ps(pixel, slope )));

        _mm_storeu_ps(out, data);

        in  += 4;
        out += 4;
    }
#else
    const float red[5] 
        = { m_red.scale,  m_red.offset,
            m_red.gamma,  m_red.breakPnt, m_red.slope };
    const float grn[5]
        = { m_green.scale, m_green.offset, 
            m_green.gamma, m_green.breakPnt, m_green.slope };
    const float blu[5]
        = { m_blue.scale,  m_blue.offset,
            m_blue.gamma,  m_blue.breakPnt, m_blue.slope };
    const float alp[5]
        = { m_alpha.scale, m_alpha.offset, 
            m_alpha.gamma, m_alpha.breakPnt, m_alpha.slope };

    for(long idx=0; idx<numPixels; ++idx)
    {
        const float pixel[4] = { in[0], in[1], in[2], in[3] };

        const float data[4] = { powf(pixel[0] * red[0] + red[1], red[2]) * m_outScale,
                                powf(pixel[1] * grn[0] + grn[1], grn[2]) * m_outScale,
                                powf(pixel[2] * blu[0] + blu[1], blu[2]) * m_outScale,
                                powf(pixel[3] * alp[0] + alp[1], alp[2]) * m_outScale };

        out[0] = pixel[0]<=red[3] ? pixel[0] * red[4] : data[0];
        out[1] = pixel[1]<=grn[3] ? pixel[1] * grn[4] : data[1];
        out[2] = pixel[2]<=blu[3] ? pixel[2] * blu[4] : data[2];
        out[3] = pixel[3]<=alp[3] ? pixel[3] * alp[4] : data[3];

        in  += 4;
        out += 4;
    }
#endif
}

GammaMoncurveOpCPURev::GammaMoncurveOpCPURev(ConstGammaOpDataRcPtr & gamma)
    :   GammaMoncurveOpCPU(gamma)
    ,   m_inScale(1.0f)
{
    update(gamma);
}

void GammaMoncurveOpCPURev::update(ConstGammaOpDataRcPtr & gamma)
{
    // NB: The power function is applied in normalized space
    // but we fold the in/out depth conversion into the other scaling
    // to minimize the number of multiplies.

    const BitDepth inBitDepth  = gamma->getInputBitDepth();
    const BitDepth outBitDepth = gamma->getOutputBitDepth();

    ComputeParamsRev(gamma->getRedParams(),   inBitDepth, outBitDepth, m_red);
    ComputeParamsRev(gamma->getGreenParams(), inBitDepth, outBitDepth, m_green);
    ComputeParamsRev(gamma->getBlueParams(),  inBitDepth, outBitDepth, m_blue);
    ComputeParamsRev(gamma->getAlphaParams(), inBitDepth, outBitDepth, m_alpha);

    m_inScale = (float)(1. / GetBitDepthMaxValue(inBitDepth));
}

void GammaMoncurveOpCPURev::apply(const void * inImg, void * outImg, long numPixels) const
{
    const float * in = (const float *)inImg;
    float * out = (float *)outImg;

#ifdef USE_SSE
    const __m128 scale
      = _mm_set_ps(m_alpha.scale, m_blue.scale,
                   m_green.scale, m_red.scale);

    const __m128 offset
      = _mm_set_ps(m_alpha.offset, m_blue.offset,
                   m_green.offset, m_red.offset);

    const __m128 gamma
      = _mm_set_ps(m_alpha.gamma, m_blue.gamma,
                   m_green.gamma, m_red.gamma);

    const __m128 breakPnt
      = _mm_set_ps(m_alpha.breakPnt, m_blue.breakPnt,
                   m_green.breakPnt, m_red.breakPnt);

    const __m128 slope
      = _mm_set_ps(m_alpha.slope, m_blue.slope,
                   m_green.slope, m_red.slope);

    const __m128 inScale = _mm_set1_ps(m_inScale);

    for(long idx=0; idx<numPixels; ++idx)
    {
        __m128 pixel = _mm_set_ps(in[3], in[2], in[1], in[0]);

        __m128 data  = _mm_mul_ps(pixel, inScale);

        data = ssePower(data, gamma);

        data = _mm_sub_ps(_mm_mul_ps(data, scale), offset);

        __m128 flag = _mm_cmpgt_ps(pixel, breakPnt);

        data = _mm_or_ps(_mm_and_ps(flag, data),
                         _mm_andnot_ps(flag, _mm_mul_ps(pixel, slope)));

        _mm_storeu_ps(out, data);

        in  += 4;
        out += 4;
    }
#else
    const float red[5] 
        = { m_red.gamma,  m_red.scale,
            m_red.offset, m_red.breakPnt, m_red.slope };
    const float grn[5]
        = { m_green.gamma, m_green.scale, 
            m_green.offset,m_green.breakPnt, m_green.slope };
    const float blu[5]
        = { m_blue.gamma,  m_blue.scale,
            m_blue.offset, m_blue.breakPnt, m_blue.slope  };
    const float alp[5]
        = { m_alpha.gamma,  m_alpha.scale, 
            m_alpha.offset, m_alpha.breakPnt, m_alpha.slope };

    for(long idx=0; idx<numPixels; ++idx)
    {
        const float pixel[4] = { in[0], in[1], in[2], in[3] };

        const float data[4] = { powf(pixel[0] * m_inScale, red[0]) * red[1] - red[2],
                                powf(pixel[1] * m_inScale, grn[0]) * grn[1] - grn[2],
                                powf(pixel[2] * m_inScale, blu[0]) * blu[1] - blu[2],
                                powf(pixel[3] * m_inScale, alp[0]) * alp[1] - alp[2] };

        out[0] = pixel[0]<=red[3] ? pixel[0] * red[4] : data[0];
        out[1] = pixel[1]<=grn[3] ? pixel[1] * grn[4] : data[1];
        out[2] = pixel[2]<=blu[3] ? pixel[2] * blu[4] : data[2];
        out[3] = pixel[3]<=alp[3] ? pixel[3] * alp[4] : data[3];

        in  += 4;
        out += 4;
    }
#endif

}

}
OCIO_NAMESPACE_EXIT