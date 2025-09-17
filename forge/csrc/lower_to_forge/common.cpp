// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "lower_to_forge/common.hpp"

#include "ops/op.hpp"
#include "utils/assert.hpp"

namespace tt
{

std::ostream &operator<<(std::ostream &os, const DataFormat &format)
{
    switch (format)
    {
        case DataFormat::Bfp2: os << "Bfp2"; break;
        case DataFormat::Bfp2_b: os << "Bfp2_b"; break;
        case DataFormat::Bfp4: os << "Bfp4"; break;
        case DataFormat::Bfp4_b: os << "Bfp4_b"; break;
        case DataFormat::Bfp8: os << "Bfp8"; break;
        case DataFormat::Bfp8_b: os << "Bfp8_b"; break;
        case DataFormat::Float16: os << "Float16"; break;
        case DataFormat::Float16_b: os << "Float16_b"; break;
        case DataFormat::Float32: os << "Float32"; break;
        case DataFormat::Int8: os << "Int8"; break;
        case DataFormat::Int32: os << "Int32"; break;
        case DataFormat::Lf8: os << "Lf8"; break;
        case DataFormat::UInt16: os << "UInt16"; break;
        case DataFormat::RawUInt8: os << "RawUInt8"; break;
        case DataFormat::RawUInt16: os << "RawUInt16"; break;
        case DataFormat::RawUInt32: os << "RawUInt32"; break;
        case DataFormat::Invalid: os << "Invalid"; break;
        default: throw std::invalid_argument("Unknown format");
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const MathFidelity &fidelity)
{
    switch (fidelity)
    {
        case MathFidelity::LoFi: os << "LoFi"; break;
        case MathFidelity::HiFi2: os << "HiFi2"; break;
        case MathFidelity::HiFi3: os << "HiFi3"; break;
        case MathFidelity::HiFi4: os << "HiFi4"; break;
        case MathFidelity::Invalid: os << "Invalid"; break;
        default: throw std::invalid_argument("Unknown fidelity");
    }
    return os;
}

MathFidelity string_to_math_fidelity(const std::string &fidelity_string)
{
    const std::unordered_map<std::string, MathFidelity> string_to_fidelity = {
        {"LoFi", MathFidelity::LoFi},
        {"HiFi2", MathFidelity::HiFi2},
        {"HiFi3", MathFidelity::HiFi3},
        {"HiFi4", MathFidelity::HiFi4}};
    auto it = string_to_fidelity.find(fidelity_string);
    TT_LOG_ASSERT(
        it != string_to_fidelity.end(), "Error: Cannot find {} in string_to_math_fidelity lookup.", fidelity_string);
    return it->second;
}

std::uint32_t data_format_byte_size(DataFormat df, int elements)
{
    switch (df)
    {
        case DataFormat::Float32: return 4 * elements;
        case DataFormat::UInt16:
        case DataFormat::Float16_b:
        case DataFormat::Float16: return 2 * elements;
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp8: return (elements + elements / 16);
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp4: return (elements / 2 + elements / 16);
        case DataFormat::Bfp2_b:
        case DataFormat::Bfp2: return (elements / 4 + elements / 16);
        case DataFormat::Lf8:
        case DataFormat::Int8: return elements;
        case DataFormat::Int32: return 4 * elements;
        case DataFormat::RawUInt8: return elements;
        case DataFormat::RawUInt16: return 2 * elements;
        case DataFormat::RawUInt32: return 4 * elements;
        case DataFormat::Invalid: return 0;
    }
    throw std::runtime_error("Invalid format");
}

MathFidelity string_to_data_format(const std::string &fidelity_string)
{
    const std::unordered_map<std::string, MathFidelity> string_to_fidelity = {
        {"LoFi", MathFidelity::LoFi},
        {"HiFi2", MathFidelity::HiFi2},
        {"HiFi3", MathFidelity::HiFi3},
        {"HiFi4", MathFidelity::HiFi4}};
    auto it = string_to_fidelity.find(fidelity_string);
    TT_LOG_ASSERT(
        it != string_to_fidelity.end(), "Error: Cannot find {} in string_to_math_fidelity lookup.", fidelity_string);
    return it->second;
}

}  // namespace tt
