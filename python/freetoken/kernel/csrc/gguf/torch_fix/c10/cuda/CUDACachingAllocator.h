#pragma once

// rpcndr.h defines `small` as `char`, conflicting with a Torch CUDA header.
#ifdef small
#undef small
#endif

#ifndef FREETOKEN_CUDA_CACHING_ALLOCATOR_HEADER
#error "FREETOKEN_CUDA_CACHING_ALLOCATOR_HEADER must name the installed Torch header"
#endif
#include FREETOKEN_CUDA_CACHING_ALLOCATOR_HEADER
