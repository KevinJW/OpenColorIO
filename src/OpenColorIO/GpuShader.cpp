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
#include <sstream>
#include <string>
#include <vector>

#include <OpenColorIO/OpenColorIO.h>

#include "DynamicProperty.h"
#include "GpuShader.h"
#include "HashUtils.h"
#include "ops/Lut3D/Lut3DOpData.h"
#include "Platform.h"


OCIO_NAMESPACE_ENTER
{

namespace
{

static void  CreateArray(const float * buf, 
                         unsigned w, unsigned h, unsigned d, 
                         GpuShaderDesc::TextureType type, 
                         std::vector<float> & res)
{
    if(buf==nullptr)
    {
        throw Exception("The buffer is invalid");
    }

    const size_t size 
        = w * h * d * (type==GpuShaderDesc::TEXTURE_RGB_CHANNEL ? 3 : 1);
    res.resize(size);
    memcpy(&res[0], buf, size * sizeof(float));
}

class PrivateImpl
{
public:
    struct Texture
    {
        Texture(const char * name, const char * identifier, 
                unsigned w, unsigned h, unsigned d,
                GpuShaderDesc::TextureType channel, 
                Interpolation interpolation, 
                const float * v)
            :   m_name(name)
            ,   m_id(identifier)
            ,   m_width(w)
            ,   m_height(h)
            ,   m_depth(d)
            ,   m_type(channel)
            ,   m_interp(interpolation)
        {
            if(!name || !*name)
            {
                throw Exception("The texture name is invalid.");
            }

            if(w==0 || h==0 || d==0)
            {
                std::stringstream ss;
                ss << "The texture buffer size is invalid: ["
                   << w << " x " << h << " x " << d << "].";

                throw Exception(ss.str().c_str());
            }

            // An unfortunate copy is mandatory to allow the creation of a GPU shader cache. 
            // The cache needs a decoupling of the processor and shader instances forbidding
            // shared naked pointer usage.
            CreateArray(v, m_width, m_height, m_depth, m_type, m_values);
        }

        std::string m_name;
        std::string m_id;
        unsigned m_width;
        unsigned m_height;
        unsigned m_depth;
        GpuShaderDesc::TextureType m_type;
        Interpolation m_interp;

        std::vector<float> m_values;

        Texture() = delete;
    };

    typedef std::vector<Texture> Textures;

    struct Uniform
    {
        Uniform(const char * name, const DynamicPropertyRcPtr & value)
            :   m_name(name)
        {
            if(!name || !*name)
            {
                throw Exception("The dynamic property name is invalid.");
            }

            if(!value->isDynamic())
            {
                // When a dynamic property is not dynamic, the GLSL fragment
                // shader program should use a constant variable 
                // (i.e. not a uniform variable).
                throw Exception("The dynamic property is not dynamic.");
            }

            m_value 
                = std::make_shared<DynamicPropertyImpl>(
                    *dynamic_cast<DynamicPropertyImpl*>(value.get()) );

        }

        std::string m_name;
        DynamicPropertyRcPtr m_value;

        Uniform() = delete;
    };

    typedef std::vector<Uniform> Uniforms;

public:
    PrivateImpl()
        :   m_max1DLUTWidth(4 * 1024)
    {
    }

    virtual ~PrivateImpl() {}

    inline unsigned get3dLutMaxDimension() const { return Lut3DOpData::maxSupportedLength; }

    inline unsigned get1dLutMaxWidth() const { return m_max1DLUTWidth; }
    inline void set1dLutMaxWidth(unsigned maxWidth) { m_max1DLUTWidth = maxWidth; }

    void addTexture(const char * name, const char * id, unsigned width, unsigned height, 
                    GpuShaderDesc::TextureType channel,
                    Interpolation interpolation, const float * values)
    {
        if(width > get1dLutMaxWidth())
        {
            std::stringstream ss;
            ss  << "1D LUT size exceeds the maximum: "
                << width << " > " << get1dLutMaxWidth();
            throw Exception(ss.str().c_str());
        }

        Texture t(name, id, width, height, 1, channel, interpolation, values);
        m_textures.push_back(t);
    }

    void getTexture(unsigned index, const char *& name, const char *& id, 
                    unsigned & width, unsigned & height, 
                    GpuShaderDesc::TextureType & channel, 
                    Interpolation & interpolation) const
    {
        if(index >= m_textures.size())
        {
            std::ostringstream ss;
            ss << "1D LUT access error: index = " << index
               << " where size = " << m_textures.size();
            throw Exception(ss.str().c_str());
        }

        const Texture & t = m_textures[index];
        name          = t.m_name.c_str();
        id            = t.m_id.c_str();
        width         = t.m_width;
        height        = t.m_height;
        channel       = t.m_type;
        interpolation = t.m_interp;
    }

    void getTextureValues(unsigned index, const float *& values) const
    {
        if(index >= m_textures.size())
        {
            std::ostringstream ss;
            ss << "1D LUT access error: index = " << index
               << " where size = " << m_textures.size();
            throw Exception(ss.str().c_str());
        }

        const Texture & t = m_textures[index];
        values   = &t.m_values[0];
    }

    void add3DTexture(const char * name, const char * id, unsigned dimension, 
                      Interpolation interpolation, const float * values)
    {
        if(dimension > get3dLutMaxDimension())
        {
            std::stringstream ss;
            ss  << "3D LUT dimension exceeds the maximum: "
                << dimension << " > " << get3dLutMaxDimension();
            throw Exception(ss.str().c_str());
        }

        Texture t(
            name, id, dimension, dimension, dimension, 
            GpuShaderDesc::TEXTURE_RGB_CHANNEL, 
            interpolation, values);
        m_textures3D.push_back(t);
    }

    void get3DTexture(unsigned index, const char *& name, const char *& id, 
                      unsigned & edgelen, 
                      Interpolation & interpolation) const
    {
        if(index >= m_textures3D.size())
        {
            std::ostringstream ss;
            ss << "3D LUT access error: index = " << index
               << " where size = " << m_textures3D.size();
            throw Exception(ss.str().c_str());
        }

        const Texture & t = m_textures3D[index];
        name          = t.m_name.c_str();
        id            = t.m_id.c_str();
        edgelen       = t.m_width;
        interpolation = t.m_interp;
    }

    void get3DTextureValues(unsigned index, const float *& values) const
    {
        if(index >= m_textures3D.size())
        {
            std::ostringstream ss;
            ss << "3D LUT access error: index = " << index
               << " where size = " << m_textures3D.size();
            throw Exception(ss.str().c_str());
        }

        const Texture & t = m_textures3D[index];
        values = &t.m_values[0];
    }

    unsigned getNumUniforms() const
    {
        return (unsigned)m_uniforms.size();
    }
    
    void getUniform(unsigned index, const char *& name, DynamicPropertyRcPtr & value) const
    {
        if (index >= (unsigned)m_uniforms.size())
        {
            std::ostringstream ss;
            ss << "Uniforms access error: index = " << index
               << " where size = " << m_uniforms.size();
            throw Exception(ss.str().c_str());
        }
        const Uniform & u = m_uniforms[index];
        name = u.m_name.c_str();
        value = u.m_value;
    }

    bool addUniform(const char * name, const DynamicPropertyRcPtr & value)
    {
        for (auto u : m_uniforms)
        {
            if (*u.m_value == *value)
            {
                if(std::string(name)!=u.m_name)
                {
                    std::string err("Same dynamic properties must have the same name: ");
                    err += u.m_name + " vs. " + name;
                    throw Exception(err.c_str());
                }

                // Uniform is already there.
                return false;
            }
        }
        m_uniforms.emplace_back(name, value);
        return true;
    }

    void createShaderText(const char * shaderDeclarations,
                          const char * shaderHelperMethods,
                          const char * shaderFunctionHeader,
                          const char * shaderFunctionBody,
                          const char * shaderFunctionFooter)
    {
        m_shaderCode.resize(0);
        m_shaderCode += (shaderDeclarations   && *shaderDeclarations)   ? shaderDeclarations   : "";
        m_shaderCode += (shaderHelperMethods  && *shaderHelperMethods)  ? shaderHelperMethods  : "";
        m_shaderCode += (shaderFunctionHeader && *shaderFunctionHeader) ? shaderFunctionHeader : "";
        m_shaderCode += (shaderFunctionBody   && *shaderFunctionBody)   ? shaderFunctionBody   : "";
        m_shaderCode += (shaderFunctionFooter && *shaderFunctionFooter) ? shaderFunctionFooter : "";

        m_shaderCodeID.resize(0);
    }

    void finalize(const std::string & cacheID)
    {
        // Finalize the shader program
        createShaderText(m_declarations.c_str(),
                         m_helperMethods.c_str(), 
                         m_functionHeader.c_str(), 
                         m_functionBody.c_str(), 
                         m_functionFooter.c_str());

        // Compute the identifier
        std::ostringstream os;
        os << m_shaderCode;
        os << "T3D: " << m_textures3D.size();
        for(auto & t : m_textures3D)
        {
            os << t.m_id << " ";
        }
        os << "T1D: " << m_textures.size();
        for(auto & t : m_textures)
        {
            os << t.m_id << " ";
        }
        os << "U: " << m_uniforms.size();
        for (auto & u : m_uniforms)
        {
            os << u.m_name << " ";
        }

        const std::string id = os.str();
        m_shaderCodeID = cacheID 
            + CacheIDHash(id.c_str(), unsigned(id.length()));
    }

    std::string m_declarations;
    std::string m_helperMethods;
    std::string m_functionHeader;
    std::string m_functionBody;
    std::string m_functionFooter;

    std::string m_shaderCode;
    std::string m_shaderCodeID;

    Textures m_textures;
    Textures m_textures3D;

    Uniforms m_uniforms;

protected:
    unsigned m_max1DLUTWidth;

private:
    PrivateImpl(const PrivateImpl & rhs) = delete;
    PrivateImpl& operator= (const PrivateImpl & rhs) = delete;
};

};

class LegacyGpuShaderDesc::Impl : public PrivateImpl
{
public:       
    explicit Impl(unsigned edgelen)
        :   PrivateImpl()
        ,   m_edgelen(edgelen)
    {
    }
    
    virtual ~Impl() {}

    inline unsigned getEdgelen() const { return m_edgelen; }

private:
    unsigned m_edgelen;
};

GpuShaderDescRcPtr LegacyGpuShaderDesc::Create(unsigned edgelen)
{
    return GpuShaderDescRcPtr(new LegacyGpuShaderDesc(edgelen), &LegacyGpuShaderDesc::Deleter);
}

LegacyGpuShaderDesc::LegacyGpuShaderDesc(unsigned edgelen)
    :   GpuShaderDesc()
    ,   m_impl(new Impl(edgelen))
{
}

LegacyGpuShaderDesc::~LegacyGpuShaderDesc()
{
    delete m_impl;
    m_impl = 0x0;
}

unsigned LegacyGpuShaderDesc::getEdgelen() const
{
    return getImpl()->getEdgelen();
}

unsigned LegacyGpuShaderDesc::getNumUniforms() const
{
    return 0;
}

void LegacyGpuShaderDesc::getUniform(unsigned, const char *&, DynamicPropertyRcPtr &) const
{
    throw Exception("Uniforms are not supported");
}

bool LegacyGpuShaderDesc::addUniform(const char *, const DynamicPropertyRcPtr &)
{
    throw Exception("Uniforms are not supported");
}

unsigned LegacyGpuShaderDesc::getTextureMaxWidth() const 
{
    throw Exception("1D LUTs are not supported");
}

void LegacyGpuShaderDesc::setTextureMaxWidth(unsigned)
{
    throw Exception("1D LUTs are not supported");
}

unsigned LegacyGpuShaderDesc::getNumTextures() const
{
    return 0;
}

void LegacyGpuShaderDesc::addTexture(
    const char *, const char *, unsigned, unsigned,
    TextureType, Interpolation, const float *)
{
    throw Exception("1D LUTs are not supported");
}

void LegacyGpuShaderDesc::getTexture(
    unsigned, const char *&, const char *&, 
    unsigned &, unsigned &, TextureType &, Interpolation &) const
{
    throw Exception("1D LUTs are not supported");
}

void LegacyGpuShaderDesc::getTextureValues(unsigned, const float *&) const
{
    throw Exception("1D LUTs are not supported");
}

unsigned LegacyGpuShaderDesc::getNum3DTextures() const
{
    return unsigned(getImpl()->m_textures3D.size());
}

void LegacyGpuShaderDesc::add3DTexture(const char * name, const char * id, unsigned dimension, 
                                       Interpolation interpolation, const float * values)
{
    if(dimension!=getImpl()->getEdgelen())
    {
        std::ostringstream ss;
        ss << "3D Texture size unexpected: " << dimension
           << " instead of " << getImpl()->getEdgelen();
        throw Exception(ss.str().c_str());
    }

    if(getImpl()->m_textures3D.size()!=0)
    {
        const std::string ss("3D Texture error: only one 3D texture allowed");
        throw Exception(ss.c_str());
    }

    getImpl()->add3DTexture(name, id, dimension, interpolation, values);
}

void LegacyGpuShaderDesc::get3DTexture(unsigned index, const char *& name, 
                                       const char *& id, unsigned & edgelen, 
                                       Interpolation & interpolation) const
{
    getImpl()->get3DTexture(index, name, id, edgelen, interpolation);
}

void LegacyGpuShaderDesc::get3DTextureValues(unsigned index, const float *& values) const
{
    getImpl()->get3DTextureValues(index, values);
}

const char * LegacyGpuShaderDesc::getShaderText() const
{
    return getImpl()->m_shaderCode.c_str();
}

const char * LegacyGpuShaderDesc::getCacheID() const
{
    return getImpl()->m_shaderCodeID.c_str();
}

void LegacyGpuShaderDesc::addToDeclareShaderCode(const char * shaderCode)
{
    if(getImpl()->m_declarations.empty())
    {
        getImpl()->m_declarations += "\n// Declaration of all variables\n\n";
    }
    getImpl()->m_declarations += (shaderCode && *shaderCode) ? shaderCode : "";
}

void LegacyGpuShaderDesc::addToHelperShaderCode(const char * shaderCode)
{
    getImpl()->m_helperMethods += (shaderCode && *shaderCode) ? shaderCode : "";
}

void LegacyGpuShaderDesc::addToFunctionShaderCode(const char * shaderCode)
{
    getImpl()->m_functionBody += (shaderCode && *shaderCode) ? shaderCode : "";
}

void LegacyGpuShaderDesc::addToFunctionHeaderShaderCode(const char * shaderCode)
{
    getImpl()->m_functionHeader += (shaderCode && *shaderCode) ? shaderCode : "";
}

void LegacyGpuShaderDesc::addToFunctionFooterShaderCode(const char * shaderCode)
{
    getImpl()->m_functionFooter += (shaderCode && *shaderCode) ? shaderCode : "";
}

void LegacyGpuShaderDesc::createShaderText(const char * shaderDeclarations, 
                                           const char * shaderHelperMethods,
                                           const char * shaderFunctionHeader,
                                           const char * shaderFunctionBody,
                                           const char * shaderFunctionFooter)
{
    getImpl()->createShaderText(shaderDeclarations,
                                shaderHelperMethods,
                                shaderFunctionHeader, 
                                shaderFunctionBody,
                                shaderFunctionFooter);
}

void LegacyGpuShaderDesc::finalize()
{
    getImpl()->finalize(GpuShaderDesc::getCacheID());
}

void LegacyGpuShaderDesc::Deleter(LegacyGpuShaderDesc* c)
{
    delete c;
}


class GenericGpuShaderDesc::Impl : public PrivateImpl
{
public:       
    Impl() : PrivateImpl() {}
    ~Impl() {}
};

GpuShaderDescRcPtr GenericGpuShaderDesc::Create()
{
    return GpuShaderDescRcPtr(new GenericGpuShaderDesc(), &GenericGpuShaderDesc::Deleter);
}

GenericGpuShaderDesc::GenericGpuShaderDesc()
    :   GpuShaderDesc()
    ,   m_impl(new Impl())
{
}

GenericGpuShaderDesc::~GenericGpuShaderDesc()
{
    delete m_impl;
    m_impl = 0x0;
}

unsigned GenericGpuShaderDesc::getNumUniforms() const
{
    return getImpl()->getNumUniforms();
}

void GenericGpuShaderDesc::getUniform(unsigned index, const char *& name,
                                      DynamicPropertyRcPtr & value) const
{
    return getImpl()->getUniform(index, name, value);
}

bool GenericGpuShaderDesc::addUniform(const char * name, const DynamicPropertyRcPtr & value)
{
    return getImpl()->addUniform(name, value);
}

unsigned GenericGpuShaderDesc::getTextureMaxWidth() const 
{
    return getImpl()->get1dLutMaxWidth();
}

void GenericGpuShaderDesc::setTextureMaxWidth(unsigned maxWidth)
{
    getImpl()->set1dLutMaxWidth(maxWidth);
}

unsigned GenericGpuShaderDesc::getNumTextures() const
{
    return unsigned(getImpl()->m_textures.size());
}

void GenericGpuShaderDesc::addTexture(const char * name, const char * id, 
                                      unsigned width, unsigned height,
                                      TextureType channel, 
                                      Interpolation interpolation,
                                      const float * values)
{
    getImpl()->addTexture(name, id, width, height, channel, interpolation, values);
}

void GenericGpuShaderDesc::getTexture(unsigned index, const char *& name, 
                                      const char *& id, unsigned & width, unsigned & height,
                                      TextureType & channel, 
                                      Interpolation & interpolation) const
{
    getImpl()->getTexture(index, name, id, width, height, channel, interpolation);
}

void GenericGpuShaderDesc::getTextureValues(unsigned index, const float *& values) const
{
    getImpl()->getTextureValues(index, values);
}

unsigned GenericGpuShaderDesc::getNum3DTextures() const
{
    return unsigned(getImpl()->m_textures3D.size());
}

void GenericGpuShaderDesc::add3DTexture(
    const char * name, const char * id, unsigned edgelen, 
    Interpolation interpolation, const float * values)
{
    getImpl()->add3DTexture(name, id, edgelen, interpolation, values);
}

void GenericGpuShaderDesc::get3DTexture(unsigned index, const char *& name, 
                                        const char *& id, unsigned & edgelen, 
                                        Interpolation & interpolation) const
{
    getImpl()->get3DTexture(index, name, id, edgelen, interpolation);
}

void GenericGpuShaderDesc::get3DTextureValues(unsigned index, const float *& values) const
{
    getImpl()->get3DTextureValues(index, values);
}

const char * GenericGpuShaderDesc::getShaderText() const
{
    return getImpl()->m_shaderCode.c_str();
}

const char * GenericGpuShaderDesc::getCacheID() const
{
    return getImpl()->m_shaderCodeID.c_str();
}

void GenericGpuShaderDesc::addToDeclareShaderCode(const char * shaderCode)
{
    if(getImpl()->m_declarations.empty())
    {
        getImpl()->m_declarations += "\n// Declaration of all variables\n\n";
    }
    getImpl()->m_declarations += (shaderCode && *shaderCode) ? shaderCode : "";
}

void GenericGpuShaderDesc::addToHelperShaderCode(const char * shaderCode)
{
    if(getImpl()->m_helperMethods.empty())
    {
        getImpl()->m_helperMethods += "\n// Declaration of all helper methods\n\n";
    }
    getImpl()->m_helperMethods += (shaderCode && *shaderCode) ? shaderCode : "";
}

void GenericGpuShaderDesc::addToFunctionShaderCode(const char * shaderCode)
{
    getImpl()->m_functionBody += (shaderCode && *shaderCode) ? shaderCode : "";
}

void GenericGpuShaderDesc::addToFunctionHeaderShaderCode(const char * shaderCode)
{
    getImpl()->m_functionHeader += (shaderCode && *shaderCode) ? shaderCode : "";
}

void GenericGpuShaderDesc::addToFunctionFooterShaderCode(const char * shaderCode)
{
    getImpl()->m_functionFooter += (shaderCode && *shaderCode) ? shaderCode : "";
}

void GenericGpuShaderDesc::createShaderText(
    const char * shaderDeclarations, const char * shaderHelperMethods,
    const char * shaderFunctionHeader, const char * shaderFunctionBody,
    const char * shaderFunctionFooter)
{
    getImpl()->createShaderText(shaderDeclarations,
                                shaderHelperMethods,
                                shaderFunctionHeader, 
                                shaderFunctionBody,
                                shaderFunctionFooter);
}

void GenericGpuShaderDesc::finalize()
{
    getImpl()->finalize(GpuShaderDesc::getCacheID());
}

void GenericGpuShaderDesc::Deleter(GenericGpuShaderDesc* c)
{
    delete c;
}

}
OCIO_NAMESPACE_EXIT


#ifdef OCIO_UNIT_TEST

namespace OCIO = OCIO_NAMESPACE;
#include "UnitTest.h"


OCIO_ADD_TEST(GpuShader, generic_shader)
{
    OCIO::GpuShaderDescRcPtr shaderDesc = OCIO::GenericGpuShaderDesc::Create();

    {
        OCIO_CHECK_NE(shaderDesc->getLanguage(), OCIO::GPU_LANGUAGE_GLSL_1_3);
        OCIO_CHECK_NO_THROW(shaderDesc->setLanguage(OCIO::GPU_LANGUAGE_GLSL_1_3));
        OCIO_CHECK_EQUAL(shaderDesc->getLanguage(), OCIO::GPU_LANGUAGE_GLSL_1_3);

        OCIO_CHECK_NE(std::string(shaderDesc->getFunctionName()), "1sd234_");
        OCIO_CHECK_NO_THROW(shaderDesc->setFunctionName("1sd234_"));
        OCIO_CHECK_EQUAL(std::string(shaderDesc->getFunctionName()), "1sd234_");

        OCIO_CHECK_NE(std::string(shaderDesc->getPixelName()), "pxl_1sd234_");
        OCIO_CHECK_NO_THROW(shaderDesc->setPixelName("pxl_1sd234_"));
        OCIO_CHECK_EQUAL(std::string(shaderDesc->getPixelName()), "pxl_1sd234_");

        OCIO_CHECK_NE(std::string(shaderDesc->getResourcePrefix()), "res_1sd234_");
        OCIO_CHECK_NO_THROW(shaderDesc->setResourcePrefix("res_1sd234_"));
        OCIO_CHECK_EQUAL(std::string(shaderDesc->getResourcePrefix()), "res_1sd234_");

        OCIO_CHECK_NO_THROW(shaderDesc->finalize());
        const std::string id(shaderDesc->getCacheID());
        OCIO_CHECK_EQUAL(id, std::string("glsl_1.3 1sd234_ res_1sd234_ pxl_1sd234_ "
                                         "$159b6ac2bdbbe3b57824faea46bd829b"));
        OCIO_CHECK_NO_THROW(shaderDesc->setResourcePrefix("res_1"));
        OCIO_CHECK_NO_THROW(shaderDesc->finalize());
        OCIO_CHECK_NE(std::string(shaderDesc->getCacheID()), id);
    }

    {
        const unsigned width  = 3;
        const unsigned height = 2;
        const unsigned size = width*height*3;

        float values[size]
            = { 0.1f, 0.2f, 0.3f,  0.4f, 0.5f, 0.6f,  0.7f, 0.8f, 0.9f,
                0.1f, 0.2f, 0.3f,  0.4f, 0.5f, 0.6f,  0.7f, 0.8f, 0.9f };

        OCIO_CHECK_EQUAL(shaderDesc->getNumTextures(), 0U);
        OCIO_CHECK_NO_THROW(shaderDesc->addTexture("lut1", "1234", width, height, 
                                                   OCIO::GpuShaderDesc::TEXTURE_RGB_CHANNEL, 
                                                   OCIO::INTERP_TETRAHEDRAL,
                                                   &values[0]));

        OCIO_CHECK_EQUAL(shaderDesc->getNumTextures(), 1U);

        const char * name = 0x0;
        const char * id = 0x0;
        unsigned w = 0;
        unsigned h = 0;
        OCIO::GpuShaderDesc::TextureType t = OCIO::GpuShaderDesc::TEXTURE_RED_CHANNEL;
        OCIO::Interpolation i = OCIO::INTERP_UNKNOWN;

        OCIO_CHECK_NO_THROW(shaderDesc->getTexture(0, name, id, w, h, t, i));

        OCIO_CHECK_EQUAL(std::string(name), "lut1");
        OCIO_CHECK_EQUAL(std::string(id), "1234");
        OCIO_CHECK_EQUAL(width, w);
        OCIO_CHECK_EQUAL(height, h);
        OCIO_CHECK_EQUAL(OCIO::GpuShaderDesc::TEXTURE_RGB_CHANNEL, t);
        OCIO_CHECK_EQUAL(OCIO::INTERP_TETRAHEDRAL, i);

        OCIO_CHECK_THROW_WHAT(shaderDesc->getTexture(1, name, id, w, h, t, i),
                              OCIO::Exception,
                              "1D LUT access error");

        const float * vals = 0x0;
        OCIO_CHECK_NO_THROW(shaderDesc->getTextureValues(0, vals));
        OCIO_CHECK_NE(vals, 0x0);
        for(unsigned idx=0; idx<size; ++idx)
        {
            OCIO_CHECK_EQUAL(values[idx], vals[idx]);
        }

        OCIO_CHECK_THROW_WHAT(shaderDesc->getTextureValues(1, vals),
                              OCIO::Exception,
                              "1D LUT access error");

        // Support several 1D LUTs

        OCIO_CHECK_NO_THROW(shaderDesc->addTexture("lut2", "1234", width, height, 
                                                   OCIO::GpuShaderDesc::TEXTURE_RGB_CHANNEL, 
                                                   OCIO::INTERP_TETRAHEDRAL,
                                                   &values[0]));

        OCIO_CHECK_EQUAL(shaderDesc->getNumTextures(), 2U);

        OCIO_CHECK_NO_THROW(shaderDesc->getTextureValues(0, vals));
        OCIO_CHECK_NO_THROW(shaderDesc->getTextureValues(1, vals));
        OCIO_CHECK_THROW_WHAT(shaderDesc->getTextureValues(2, vals),
                              OCIO::Exception,
                              "1D LUT access error");
    }

    {
        const unsigned edgelen = 2;
        const unsigned size = edgelen*edgelen*edgelen*3;
        float values[size]
            = { 0.1f, 0.2f, 0.3f,  0.4f, 0.5f, 0.6f,  0.7f, 0.8f, 0.9f,  0.7f, 0.8f, 0.9f,
                0.1f, 0.2f, 0.3f,  0.4f, 0.5f, 0.6f,  0.7f, 0.8f, 0.9f,  0.7f, 0.8f, 0.9f, };

        OCIO_CHECK_EQUAL(shaderDesc->getNum3DTextures(), 0U);
        OCIO_CHECK_NO_THROW(shaderDesc->add3DTexture("lut1", "1234", edgelen, 
                                                     OCIO::INTERP_TETRAHEDRAL,
                                                     &values[0]));

        OCIO_CHECK_EQUAL(shaderDesc->getNum3DTextures(), 1U);

        const char * name = 0x0;
        const char * id = 0x0;
        unsigned e = 0;
        OCIO::Interpolation i = OCIO::INTERP_UNKNOWN;

        OCIO_CHECK_NO_THROW(shaderDesc->get3DTexture(0, name, id, e, i));

        OCIO_CHECK_EQUAL(std::string(name), "lut1");
        OCIO_CHECK_EQUAL(std::string(id), "1234");
        OCIO_CHECK_EQUAL(edgelen, e);
        OCIO_CHECK_EQUAL(OCIO::INTERP_TETRAHEDRAL, i);

        OCIO_CHECK_THROW_WHAT(shaderDesc->get3DTexture(1, name, id, e, i),
                              OCIO::Exception,
                              "3D LUT access error");

        const float * vals = 0x0;
        OCIO_CHECK_NO_THROW(shaderDesc->get3DTextureValues(0, vals));
        OCIO_CHECK_NE(vals, 0x0);
        for(unsigned idx=0; idx<size; ++idx)
        {
            OCIO_CHECK_EQUAL(values[idx], vals[idx]);
        }

        OCIO_CHECK_THROW_WHAT(shaderDesc->get3DTextureValues(1, vals),
                              OCIO::Exception,
                              "3D LUT access error");

        // Supports several 3D LUTs

        OCIO_CHECK_NO_THROW(shaderDesc->add3DTexture("lut1", "1234", edgelen, 
                                                     OCIO::INTERP_TETRAHEDRAL,
                                                     &values[0]));

        OCIO_CHECK_EQUAL(shaderDesc->getNum3DTextures(), 2U);

        // Check the 3D LUT limit

        OCIO_CHECK_THROW(shaderDesc->add3DTexture("lut1", "1234", 130,
                                                  OCIO::INTERP_TETRAHEDRAL,
                                                  &values[0]),
                         OCIO::Exception);
    }

    {
        OCIO_CHECK_NO_THROW(shaderDesc->addToDeclareShaderCode("vec2 coords;\n"));
        OCIO_CHECK_NO_THROW(shaderDesc->addToHelperShaderCode("vec2 helpers() {}\n\n"));
        OCIO_CHECK_NO_THROW(shaderDesc->addToFunctionHeaderShaderCode("void func() {\n"));
        OCIO_CHECK_NO_THROW(shaderDesc->addToFunctionShaderCode("  int i;\n"));
        OCIO_CHECK_NO_THROW(shaderDesc->addToFunctionFooterShaderCode("}\n"));

        OCIO_CHECK_NO_THROW(shaderDesc->finalize());

        std::string fragText;
        fragText += "\n";
        fragText += "// Declaration of all variables\n";
        fragText += "\n";
        fragText += "vec2 coords;\n";
        fragText += "\n";
        fragText += "// Declaration of all helper methods\n";
        fragText += "\n";
        fragText += "vec2 helpers() {}\n\n";
        fragText += "void func() {\n";
        fragText += "  int i;\n";
        fragText += "}\n";

        OCIO_CHECK_EQUAL(fragText, shaderDesc->getShaderText());
    }
}

OCIO_ADD_TEST(GpuShader, legacy_shader)
{
    const unsigned edgelen = 2;

    OCIO::GpuShaderDescRcPtr shaderDesc = OCIO::LegacyGpuShaderDesc::Create(edgelen);

    {
        const unsigned width  = 3;
        const unsigned height = 2;
        const unsigned size = width*height*3;
        float values[size];

        OCIO_CHECK_EQUAL(shaderDesc->getNumTextures(), 0U);
        OCIO_CHECK_THROW_WHAT(shaderDesc->addTexture("lut1", "1234", width, height, 
                                                     OCIO::GpuShaderDesc::TEXTURE_RGB_CHANNEL, 
                                                     OCIO::INTERP_TETRAHEDRAL,
                                                     &values[0]), 
                              OCIO::Exception,
                              "1D LUTs are not supported");

        const char * name = 0x0;
        const char * id = 0x0;
        unsigned w = 0;
        unsigned h = 0;
        OCIO::GpuShaderDesc::TextureType t = OCIO::GpuShaderDesc::TEXTURE_RED_CHANNEL;
        OCIO::Interpolation i = OCIO::INTERP_UNKNOWN;

        OCIO_CHECK_THROW_WHAT(shaderDesc->getTexture(0, name, id, w, h, t, i),
                              OCIO::Exception,
                              "1D LUTs are not supported");
    }

    {
        const unsigned size = edgelen*edgelen*edgelen*3;
        float values[size]
            = { 0.1f, 0.2f, 0.3f,  0.4f, 0.5f, 0.6f,  0.7f, 0.8f, 0.9f,  0.7f, 0.8f, 0.9f,
                0.1f, 0.2f, 0.3f,  0.4f, 0.5f, 0.6f,  0.7f, 0.8f, 0.9f,  0.7f, 0.8f, 0.9f, };

        OCIO_CHECK_EQUAL(shaderDesc->getNum3DTextures(), 0U);
        OCIO_CHECK_NO_THROW(shaderDesc->add3DTexture("lut1", "1234", edgelen, 
                                                     OCIO::INTERP_TETRAHEDRAL,
                                                     &values[0]));

        OCIO_CHECK_EQUAL(shaderDesc->getNum3DTextures(), 1U);

        const char * name = 0x0;
        const char * id = 0x0;
        unsigned e = 0;
        OCIO::Interpolation i = OCIO::INTERP_UNKNOWN;

        OCIO_CHECK_NO_THROW(shaderDesc->get3DTexture(0, name, id, e, i));

        OCIO_CHECK_EQUAL(std::string(name), "lut1");
        OCIO_CHECK_EQUAL(std::string(id), "1234");
        OCIO_CHECK_EQUAL(edgelen, e);
        OCIO_CHECK_EQUAL(OCIO::INTERP_TETRAHEDRAL, i);

        OCIO_CHECK_THROW_WHAT(shaderDesc->get3DTexture(1, name, id, e, i),
                              OCIO::Exception,
                              "3D LUT access error");

        const float * vals = 0x0;
        OCIO_CHECK_NO_THROW(shaderDesc->get3DTextureValues(0, vals));
        OCIO_CHECK_NE(vals, 0x0);
        for(unsigned idx=0; idx<size; ++idx)
        {
            OCIO_CHECK_EQUAL(values[idx], vals[idx]);
        }

        OCIO_CHECK_THROW_WHAT(shaderDesc->get3DTextureValues(1, vals),
                              OCIO::Exception,
                              "3D LUT access error");

        // Only one 3D LUT

        OCIO_CHECK_THROW_WHAT(shaderDesc->add3DTexture("lut1", "1234", edgelen, 
                                                       OCIO::INTERP_TETRAHEDRAL,
                                                       &values[0]),
                              OCIO::Exception,
                              "only one 3D texture allowed");

    }
}


#endif