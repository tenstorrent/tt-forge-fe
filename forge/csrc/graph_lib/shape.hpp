// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ostream>
#include <vector>

#include "lower_to_forge/common.hpp"
#include "nlohmann/json.hpp"
#include "utils/assert.hpp"
using json = nlohmann::json;

namespace tt
{

namespace graphlib
{

using DimBroadcast = std::tuple<int, int, int>;  // operand, dim, size

class Shape
{
   public:
    enum Type
    {
        FREE,  // any number of dimensions
        FORGE  // 4D, snapped to tile sizes
    };

   private:
    bool valid_ = false;
    Shape::Type type_ = FREE;
    std::vector<std::uint32_t> dims_;
    TileDim tile_dim_ = TileDim::Dim32x32;

   public:
    constexpr static int FORGE_TILE_DIM = 32;
    constexpr static int FORGE_DIM_COUNT = 4;
    constexpr static int FORGE_MAX_DIM_COUNT = 5;

    Shape() = default;
    Shape(bool valid, Shape::Type type, std::vector<std::uint32_t> dims);

    template <class T = std::uint32_t>
    static Shape create(std::vector<T> dims)
    {
        return Shape(true, FREE, std::vector<std::uint32_t>(dims.begin(), dims.end()));
    }
    static Shape create_forge(
        std::vector<std::uint32_t> dims, int tile_height = FORGE_TILE_DIM, int tile_width = FORGE_TILE_DIM);
    static Shape create_forge(std::uint32_t w, std::uint32_t z, std::uint32_t r, std::uint32_t c);
    static Shape create_with_type_from_other(const Shape &other, std::vector<std::uint32_t> dims);
    static Shape to_forge(const Shape &other);

    std::vector<std::uint32_t>::iterator begin() { return dims_.begin(); }
    std::vector<std::uint32_t>::iterator end() { return dims_.end(); }

    std::uint32_t &operator[](int i);
    std::uint32_t const &operator[](int i) const;

    TileDim get_tile_dim() const { return tile_dim_; }
    void set_tile_dim(TileDim tile_dim) { this->tile_dim_ = tile_dim; }
    int get_tile_height() const;
    int get_tile_width() const;
    int get_tile_volume() const { return get_tile_height() * get_tile_width(); }

    bool operator==(const Shape &other) const;
    bool operator!=(const Shape &other) const;

    bool is_valid() const { return valid_; }
    bool is_forge() const { return type_ == FORGE; }
    Shape::Type type() const { return type_; }

    template <class T = uint32_t>
    std::vector<T> as_vector() const
    {
        return std::vector<T>(dims_.begin(), dims_.end());
    }

    std::tuple<int, int, int, int> as_tuple() const;
    std::string as_string() const;

    std::uint32_t size() const { return (std::uint32_t)dims_.size(); }
    std::uint32_t volume() const;
    int negative_index(int index) const { return (index < 0) ? index : (index - (int)size()); }
    int positive_index(int index) const { return (index < 0) ? (index + (int)size()) : index; }
    bool index_in_bounds(int index) const { return positive_index(index) >= 0 and positive_index(index) < (int)size(); }
    // If forge shape, returns true if single tile for R/C dims
    // otherwise returns true if provided dim == 1
    bool is_unit(int index) const;
    bool is_single_tile() const;

    // Return a canonical copy, i.e. padded to 4d
    Shape canonical() const;
    Shape as_rank(std::uint32_t rank) const;

    // Return the list of dims (and amount) that need to be broadcast from current to other
    std::vector<DimBroadcast> broadcast_dims(const Shape &other) const;

    // Common factory func
    static Shape single_tile() { return create_forge(1, 1, FORGE_TILE_DIM, FORGE_TILE_DIM); }

    // Forge dimensions accessors
    std::uint32_t rt() const;
    std::uint32_t ct() const;
    std::uint32_t z() const;
    std::uint32_t w() const;

    // Pickling methods
    std::tuple<bool, int, std::vector<std::uint32_t>> get_pickle_data() const;
    static Shape create_from_pickled(bool valid, int type, std::vector<std::uint32_t> dims);

    // Macro used to generate the to_json/from_json methods for each shape type
    NLOHMANN_JSON_SERIALIZE_ENUM(Shape::Type, {{Shape::Type::FREE, "FREE"}, {Shape::Type::FORGE, "FORGE"}});
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Shape, valid_, type_, dims_)
};

std::ostream &operator<<(std::ostream &out, const Shape &s);

template <typename T>
inline T align_up_tile(T d)
{
    d -= 1;
    return static_cast<T>(d - (d % graphlib::Shape::FORGE_TILE_DIM) + graphlib::Shape::FORGE_TILE_DIM);
}

}  // namespace graphlib
}  // namespace tt
