/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <cstdint>

namespace vllm_ascend {

constexpr uint64_t MOE_LORA_PREFILL_UB_ALIGNMENT = 32U;
constexpr uint32_t MOE_LORA_PREFILL_ERROR_RECORD_LANES = 8U;

struct MoeLoraPrefillTilingInput {
    uint64_t ub_size;
    uint32_t num_groups;
    uint32_t group_pitch;
    uint32_t num_cores;
    uint32_t index_or_count_bytes;
    uint32_t enabled_bytes;
    uint32_t data_bytes;
    uint32_t route_extent;
    uint32_t input_width;
    uint32_t delta_width;
};

struct MoeLoraPrefillUbUsage {
    uint64_t route_allgather;
    uint64_t route_alltoall;
    uint64_t prefix_b1;
    uint64_t prefix_b2;
    uint64_t scatter_allgather;
    uint64_t scatter_alltoall;
    uint64_t gather_by_perm;
    uint64_t scatter_add;
};

struct MoeLoraPrefillHostTiling {
    uint32_t route_tile_rows;
    uint32_t prefix_tile_groups;
    uint32_t column_tile_elements;
    uint32_t scatter_add_tile_elements;
    uint64_t ub_size;
    uint64_t usable_ub_size;
    MoeLoraPrefillUbUsage usage;
    bool valid;
};

inline uint64_t moe_lora_prefill_align_up_ub(uint64_t bytes)
{
    return (bytes + MOE_LORA_PREFILL_UB_ALIGNMENT - 1U) /
        MOE_LORA_PREFILL_UB_ALIGNMENT * MOE_LORA_PREFILL_UB_ALIGNMENT;
}

inline uint64_t moe_lora_prefill_queue_bytes(uint32_t buffer_num,
                                             uint64_t bytes_per_buffer)
{
    // InitBuffer(queue, num, size) allocates `num` physical slots. TQue's
    // template depth is an event/queue property and does not multiply UB.
    return static_cast<uint64_t>(buffer_num) *
        moe_lora_prefill_align_up_ub(bytes_per_buffer);
}

inline MoeLoraPrefillUbUsage moe_lora_prefill_calculate_ub_usage(
    const MoeLoraPrefillTilingInput& input, uint32_t route_tile_rows,
    uint32_t prefix_tile_groups, uint32_t column_tile_elements,
    uint32_t scatter_add_tile_elements)
{
    constexpr uint64_t int32_bytes = sizeof(int32_t);
    constexpr uint64_t int64_bytes = sizeof(int64_t);
    constexpr uint64_t record_bytes =
        MOE_LORA_PREFILL_ERROR_RECORD_LANES * int32_bytes;
    const uint64_t group_pitch = input.group_pitch;
    const uint64_t route_rows = route_tile_rows;
    const uint64_t prefix_pitch_groups =
        (static_cast<uint64_t>(prefix_tile_groups) + 7U) / 8U * 8U;
    const uint64_t columns = column_tile_elements;
    const uint64_t scatter_elements = scatter_add_tile_elements;

    MoeLoraPrefillUbUsage usage{};
    usage.route_allgather =
        2U * moe_lora_prefill_queue_bytes(
                 1U, route_rows * input.index_or_count_bytes) +
        moe_lora_prefill_queue_bytes(1U, route_rows * int64_bytes) +
        moe_lora_prefill_queue_bytes(1U, group_pitch * input.enabled_bytes) +
        moe_lora_prefill_queue_bytes(
            1U, (group_pitch + MOE_LORA_PREFILL_ERROR_RECORD_LANES) *
                    int32_bytes);

    usage.route_alltoall =
        moe_lora_prefill_queue_bytes(
            1U, group_pitch * input.index_or_count_bytes) +
        moe_lora_prefill_queue_bytes(1U, route_rows * int64_bytes) +
        moe_lora_prefill_queue_bytes(1U, group_pitch * input.enabled_bytes) +
        moe_lora_prefill_queue_bytes(
            1U, (group_pitch + MOE_LORA_PREFILL_ERROR_RECORD_LANES) *
                    int32_bytes);

    usage.prefix_b1 =
        2U * moe_lora_prefill_queue_bytes(
                 1U, static_cast<uint64_t>(input.num_cores) *
                         prefix_pitch_groups * int32_bytes) +
        moe_lora_prefill_queue_bytes(
            1U, prefix_pitch_groups * int32_bytes);

    usage.prefix_b2 =
        moe_lora_prefill_queue_bytes(1U, group_pitch * int32_bytes) +
        moe_lora_prefill_queue_bytes(
            1U, static_cast<uint64_t>(input.num_cores) * record_bytes) +
        moe_lora_prefill_queue_bytes(1U, group_pitch * int32_bytes) +
        moe_lora_prefill_queue_bytes(1U, group_pitch * int64_bytes) +
        moe_lora_prefill_queue_bytes(1U, record_bytes);

    const uint64_t double_buffered_io =
        2U * moe_lora_prefill_queue_bytes(
                 2U, columns * input.data_bytes);
    const uint64_t double_buffered_single_record =
        moe_lora_prefill_queue_bytes(2U, record_bytes);
    usage.scatter_allgather =
        moe_lora_prefill_queue_bytes(1U, 4U * group_pitch * int32_bytes) +
        moe_lora_prefill_queue_bytes(1U, group_pitch * input.enabled_bytes) +
        2U * moe_lora_prefill_queue_bytes(
                 1U, route_rows * input.index_or_count_bytes) +
        moe_lora_prefill_queue_bytes(1U, route_rows * int64_bytes) +
        double_buffered_io + double_buffered_single_record;

    usage.scatter_alltoall =
        moe_lora_prefill_queue_bytes(1U, 4U * group_pitch * int32_bytes) +
        moe_lora_prefill_queue_bytes(
            1U, group_pitch * input.index_or_count_bytes) +
        moe_lora_prefill_queue_bytes(1U, route_rows * int64_bytes) +
        moe_lora_prefill_queue_bytes(1U, group_pitch * input.enabled_bytes) +
        double_buffered_io + double_buffered_single_record;

    usage.gather_by_perm =
        moe_lora_prefill_queue_bytes(1U, route_rows * record_bytes) +
        double_buffered_io;

    usage.scatter_add =
        moe_lora_prefill_queue_bytes(1U, route_rows * record_bytes) +
        3U * moe_lora_prefill_queue_bytes(
                 2U, scatter_elements * input.data_bytes) +
        2U * moe_lora_prefill_queue_bytes(
                 1U, scatter_elements * sizeof(float));
    return usage;
}

inline bool moe_lora_prefill_usage_fits(const MoeLoraPrefillUbUsage& usage,
                                        uint64_t usable)
{
    return usage.route_allgather <= usable &&
        usage.route_alltoall <= usable && usage.prefix_b1 <= usable &&
        usage.prefix_b2 <= usable && usage.scatter_allgather <= usable &&
        usage.scatter_alltoall <= usable && usage.gather_by_perm <= usable &&
        usage.scatter_add <= usable;
}

template <typename Fits>
inline uint32_t moe_lora_prefill_max_fitting_extent(uint32_t limit, Fits fits)
{
    if (limit == 0U || !fits(1U)) {
        return 0U;
    }
    uint32_t low = 1U;
    uint32_t high = limit;
    while (low < high) {
        const uint32_t mid = low +
            static_cast<uint32_t>((static_cast<uint64_t>(high) - low + 1U) / 2U);
        if (fits(mid)) {
            low = mid;
        } else {
            high = mid - 1U;
        }
    }
    return low;
}

inline uint32_t moe_lora_prefill_clamp_aligned_extent(
    uint32_t actual_extent, uint32_t calculated_extent, uint32_t alignment)
{
    const uint32_t limited = std::min(actual_extent, calculated_extent);
    if (limited < alignment) {
        return limited;
    }
    return limited / alignment * alignment;
}

inline MoeLoraPrefillHostTiling make_moe_lora_prefill_host_tiling(
    const MoeLoraPrefillTilingInput& input)
{
    MoeLoraPrefillHostTiling plan{};
    plan.ub_size = input.ub_size;
    if (input.ub_size < 32U * 1024U || input.num_groups == 0U ||
        input.group_pitch < input.num_groups ||
        input.num_cores == 0U || input.index_or_count_bytes == 0U ||
        input.enabled_bytes == 0U || input.data_bytes == 0U ||
        input.route_extent == 0U || input.input_width == 0U ||
        input.delta_width == 0U) {
        return plan;
    }

    // Preserve one eighth for TPipe bookkeeping and implementation-specific
    // alignment. Every reported live set below must fit the smaller budget.
    plan.usable_ub_size =
        (input.ub_size * 7U / 8U) / MOE_LORA_PREFILL_UB_ALIGNMENT *
        MOE_LORA_PREFILL_UB_ALIGNMENT;
    const uint32_t rows_per_core =
        static_cast<uint32_t>((static_cast<uint64_t>(input.route_extent) +
                               input.num_cores - 1U) /
                              input.num_cores);

    const uint32_t max_route_rows = moe_lora_prefill_max_fitting_extent(
        rows_per_core, [&](uint32_t rows) {
            const auto usage = moe_lora_prefill_calculate_ub_usage(
                input, rows, 1U, 1U, 1U);
            return usage.route_allgather <= plan.usable_ub_size &&
                usage.route_alltoall <= plan.usable_ub_size &&
                usage.scatter_allgather <= plan.usable_ub_size &&
                usage.scatter_alltoall <= plan.usable_ub_size &&
                usage.gather_by_perm <= plan.usable_ub_size &&
                usage.scatter_add <= plan.usable_ub_size;
        });
    plan.route_tile_rows = moe_lora_prefill_clamp_aligned_extent(
        rows_per_core, max_route_rows, 8U);
    if (plan.route_tile_rows == 0U) {
        return plan;
    }

    const uint32_t max_prefix_groups = moe_lora_prefill_max_fitting_extent(
        input.num_groups, [&](uint32_t groups) {
            const auto usage = moe_lora_prefill_calculate_ub_usage(
                input, plan.route_tile_rows, groups, 1U, 1U);
            return usage.prefix_b1 <= plan.usable_ub_size;
        });
    plan.prefix_tile_groups = moe_lora_prefill_clamp_aligned_extent(
        input.num_groups, max_prefix_groups, 8U);
    if (plan.prefix_tile_groups == 0U) {
        return plan;
    }

    const uint32_t max_columns = moe_lora_prefill_max_fitting_extent(
        input.input_width, [&](uint32_t columns) {
            const auto usage = moe_lora_prefill_calculate_ub_usage(
                input, plan.route_tile_rows, plan.prefix_tile_groups,
                columns, 1U);
            return usage.scatter_allgather <= plan.usable_ub_size &&
                usage.scatter_alltoall <= plan.usable_ub_size &&
                usage.gather_by_perm <= plan.usable_ub_size;
        });
    plan.column_tile_elements = moe_lora_prefill_clamp_aligned_extent(
        input.input_width, max_columns, 16U);
    if (plan.column_tile_elements == 0U) {
        return plan;
    }

    const uint32_t max_scatter_elements = moe_lora_prefill_max_fitting_extent(
        input.delta_width, [&](uint32_t elements) {
            const auto usage = moe_lora_prefill_calculate_ub_usage(
                input, plan.route_tile_rows, plan.prefix_tile_groups,
                plan.column_tile_elements, elements);
            return usage.scatter_add <= plan.usable_ub_size;
        });
    plan.scatter_add_tile_elements = moe_lora_prefill_clamp_aligned_extent(
        input.delta_width, max_scatter_elements, 16U);
    if (plan.scatter_add_tile_elements == 0U) {
        return plan;
    }

    plan.usage = moe_lora_prefill_calculate_ub_usage(
        input, plan.route_tile_rows, plan.prefix_tile_groups,
        plan.column_tile_elements, plan.scatter_add_tile_elements);
    plan.valid = moe_lora_prefill_usage_fits(plan.usage, plan.usable_ub_size);
    return plan;
}

}  // namespace vllm_ascend
