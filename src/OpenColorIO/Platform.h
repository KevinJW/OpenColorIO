/*
Copyright (c) 2003-2010 Sony Pictures Imageworks Inc., et al.
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


#ifndef INCLUDED_OCIO_PLATFORM_H
#define INCLUDED_OCIO_PLATFORM_H

// platform-specific includes
#if defined(_WIN32)

#define _CRT_NONSTDC_NO_DEPRECATE 1
#define _CRT_SECURE_NO_DEPRECATE 1
#define NOMINMAX 1

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// windows - defined for both Win32 and Win64

#include <windows.h>

#include <malloc.h>
#include <io.h>
#include <tchar.h> 
#include <process.h>

#else
// assume linux/unix/posix

#include <stdlib.h>
#if !defined(__FreeBSD__)
#include <alloca.h>
#endif
#include <string.h>
#include <pthread.h>

#endif // defined(_WIN32)

// general includes
#include <stdio.h>
#include <math.h>
#include <assert.h>

// missing functions on Windows
#ifdef _WIN32
#define snprintf sprintf_s
#define strtok_r strtok_s
#define sscanf sscanf_s
#define putenv _putenv
typedef __int64 FilePos;
#define fseeko _fseeki64
#define ftello _ftelli64

inline double log2(double x)
{
    return log(x) * 1.4426950408889634; 
}

#else
typedef off_t FilePos;
#endif // _WIN32
    

OCIO_NAMESPACE_ENTER
{

// TODO: Add proper endian detection using architecture / compiler mojo
//       In the meantime, hardcode to x86
#define OCIO_LITTLE_ENDIAN 1  // This is correct on x86

namespace Platform
{

void Getenv (const char* name, std::string& value);

// Case insensitive string comparison
int Strcasecmp(const char* str1, const char* str2);

// Case insensitive string comparison for the nth first characters only
int Strncasecmp(const char* str1, const char* str2, size_t n);

// Allocates memory on a specified alignment boundary. Must use
// AlignedFree to free the memory block.
// An exception is thrown if an allocation error occurs.
void* AlignedMalloc(size_t size, size_t alignment);

// Frees a block of memory that was allocated with AlignedMalloc.
void AlignedFree(void* memBlock);

// Create a temporary filename where filenameExt could be empty.
void CreateTempFilename(std::string & filename, const std::string & filenameExt);

}

}
OCIO_NAMESPACE_EXIT

#endif // INCLUDED_OCIO_PLATFORM_H
