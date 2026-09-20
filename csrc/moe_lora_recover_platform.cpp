// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

// SDK-only translation unit. CANN platform uses the old libstdc++ string ABI;
// Torch bindings retain their own ABI and cross this boundary through uint64_t.
#include <cstdint>
#include <acl/acl_rt.h>
#include <platform/platform_info.h>

extern "C" uint64_t vllm_ascend_recover_ub_bytes()
{
    const char* soc = aclrtGetSocName();
    if (soc == nullptr) {
        return 0;
    }
    auto& manager = fe::PlatformInfoManager::Instance();
    if (manager.InitializePlatformInfo() != 0) {
        return 0;
    }
    fe::PlatFormInfos info;
    fe::OptionalInfos optional;
    if (manager.GetPlatformInfos(soc, info, optional) != 0) {
        return 0;
    }
    uint64_t bytes = 0;
    info.GetLocalMemSize(fe::LocalMemType::UB, bytes);
    return bytes;
}
