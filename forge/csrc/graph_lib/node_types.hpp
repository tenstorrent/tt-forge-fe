// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utils/assert.hpp>
#include <variant>
#include <vector>

#include "graph_lib/defines.hpp"
#include "graph_lib/edge.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/shape.hpp"
#include "graph_lib/utils.hpp"
#include "lower_to_forge/common.hpp"
#include "ops/op.hpp"
#include "shared_utils/sparse_matmul_utils.hpp"

namespace tt
{
class FusedOp;

enum class DecomposeEpoch : uint8_t;
class DecomposingContext;

namespace graphlib
{

bool is_permute_xy_order(const std::vector<int> &order);
std::vector<int> create_permute_xy_order(const int rank);

// fwd declares
class Graph;
class ConstEvalGraph;

// Simple create(..) function under the hood just constructs a unique-ptr of the specified type with arguments forwarded
// and returns the object.
template <typename ClassType, typename... ClassArgs>
std::unique_ptr<ClassType> create_node(ClassArgs &&...args)
{
    return std::make_unique<ClassType>(std::forward<ClassArgs>(args)...);
}

enum QueueNodeType
{
    Input,
    Output,
    EpochToEpoch,
    GradAccumulator,
    Buffering,
};

enum MemoryAccessType
{
    FIFO,
    RAM,
};

// TaggedNode is just a wrapper around Node class and adds tag attributes
// so we can freely annotate and have a generic interface for passing
// hints across all node-types and compiler passes
using TagKey = std::string;
using TagValue = std::variant<bool, std::uint32_t, std::string>;
using TagHints = std::map<TagKey, TagValue>;

struct TaggedNode : public Node
{
    TagHints hints;

    bool has_tag(const std::string &tag) const;
    TagValue tag_value(const std::string &tag) const;
    template <typename T>
    T tag_value(const std::string &tag) const
    {
        return std::get<T>(tag_value(tag));
    }
    template <typename T>
    T tag_value_or(const std::string &tag, T def = T{}) const
    {
        if (has_tag(tag))
            return std::get<T>(tag_value(tag));
        else
            return def;
    }
    const TagHints &get_tags() const;

    // default: tagging without a tag_value just sets membership of tag_value as true
    void tag(const std::string &tag, const TagValue &tag_value = true);
    void add_tags(const TagHints &other_tags);

    TaggedNode(std::string name, NodeType node_type) : Node(name, node_type) {}
};

class QueueNode : public TaggedNode
{
   protected:
    QueueNodeType queue_type_;
    MemoryAccessType memory_access_type_;
    int entries_;
    QueueNode *loopback_queue_ = nullptr;
    std::string alias_;

   public:
    QueueNode(const std::string &name, QueueNodeType queue_type, NodeType node_type = NodeType::kQueue) :
        TaggedNode(name, node_type), queue_type_(queue_type), memory_access_type_(MemoryAccessType::FIFO), entries_(0)
    {
    }
    std::string queue_type_string() const;

    virtual std::unique_ptr<Node> clone(std::string const &name = "") const override;

    void set_num_entries(int entries) { entries_ = entries; }
    int get_num_entries() const { return entries_; }
    std::string get_alias() const { return alias_; }
    void set_alias(std::string alias) { alias_ = alias; }

    bool is_epoch_to_epoch() const { return queue_type_ == EpochToEpoch; }
    bool is_grad_accumulator() const { return queue_type_ == GradAccumulator; }
    bool is_input() const { return queue_type_ == Input; }
    bool is_output() const { return queue_type_ == Output; }
    bool is_buffering() const { return queue_type_ == Buffering; }

    QueueNodeType queue_type() const { return this->queue_type_; }
    void set_queue_type(QueueNodeType queue_type) { this->queue_type_ = queue_type; }

    MemoryAccessType memory_access_type() const { return this->memory_access_type_; }
    void set_memory_access_type(MemoryAccessType memory_access_type) { this->memory_access_type_ = memory_access_type; }
    std::string memory_access_type_string() const;
};

class EpochToEpochQueueNode : public QueueNode
{
   protected:
    bool cross_epoch_type_;  // it's used between two epochs that are not of the same type (usually fwd->bwd)
    bool cross_chip_type_;   // it's used between two chips
   public:
    EpochToEpochQueueNode(const std::string &name, bool cross_epoch_type, bool cross_chip_type) :
        QueueNode(name, QueueNodeType::EpochToEpoch),
        cross_epoch_type_(cross_epoch_type),
        cross_chip_type_(cross_chip_type)
    {
    }
    virtual std::unique_ptr<Node> clone(std::string const &name = "") const override;

    bool is_cross_epoch_type() const { return cross_epoch_type_; }
    bool is_cross_chip_type() const { return cross_chip_type_; }
};

class BufferingQueueNode : public QueueNode
{
   public:
    BufferingQueueNode(const std::string &name, int num_entries) : QueueNode(name, QueueNodeType::Buffering)
    {
        this->set_num_entries(num_entries);
    }
    virtual std::unique_ptr<Node> clone(std::string const &name = "") const override;
};

enum InputNodeType
{
    Accumulator,
    Activation,
    Constant,
    Gradient,
    Loss,
    OptimizerParameter,
    Parameter,
    Target,
};

class InputNode : public QueueNode
{
   private:
    InputNodeType input_type_;
    bool requires_grad_;
    std::vector<int> tile_broadcast_dims_;
    bool prologue_ = false;
    std::string fractured_parameter_mapping_;
    RuntimeTensorTransform runtime_tensor_transform;

   protected:
    std::unique_ptr<ConstEvalGraph> consteval_graph_;

   public:
    InputNode(const std::string &name, InputNodeType input_type, bool requires_grad);
    virtual ~InputNode();

    InputNodeType input_type() const { return input_type_; }
    std::string input_type_string() const;
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    void clone_consteval_graph_from(Node *original);
    ConstEvalGraph *get_consteval_graph(Graph *graph = nullptr, bool create = false, bool promote_input = false);
    void clear_consteval_graph();
    virtual std::unique_ptr<Node> clone(std::string const &name = "") const override;

    void set_tile_broadcast_dim(int dim) { tile_broadcast_dims_.push_back(dim); }
    std::vector<int> get_tile_broadcast_dims() const { return tile_broadcast_dims_; }

    void set_prologue(bool prologue) { prologue_ = prologue; }
    bool is_prologue() const { return prologue_; }
    void set_fractured_parameter_mapping(std::string name) { fractured_parameter_mapping_ = name; }
    std::string get_fractured_parameter_mapping() const { return fractured_parameter_mapping_; }

    void set_runtime_tensor_transform(RuntimeTensorTransform transform) { this->runtime_tensor_transform = transform; }

    RuntimeTensorTransform &get_runtime_tensor_transform() { return this->runtime_tensor_transform; }

    bool is_constant() const { return input_type_ == Constant; }
    bool is_parameter() const { return input_type_ == Parameter; }
    bool is_gradient() const { return input_type_ == Gradient; }
    bool is_loss() const { return input_type_ == Loss; }
    bool is_target() const { return input_type_ == Target; }
    bool is_accumulator() const { return input_type_ == Accumulator; }
    bool is_activation() const { return input_type_ == Activation; }
    bool is_optimizer_parameter() const { return input_type_ == OptimizerParameter; }
};

enum ConstantInputNodeType
{
    SingleValue,
    Tensor,
};

class ConstantInputNode : public InputNode
{
   private:
    ConstantInputNodeType node_type_;
    float constant_value_;
    std::shared_ptr<void> tensor_handle_;
    Shape tensor_shape_;
    sparse::SparseFORGE sparse_forge{};  // TODO: This should be a void pointer

    int dim_r_;
    int dim_c_;

   public:
    ConstantInputNode(const std::string &name, float constant_value, int dim_r = -1, int dim_c = -1) :
        InputNode(name, InputNodeType::Constant, NodeType::kInput),
        node_type_(ConstantInputNodeType::SingleValue),
        constant_value_(constant_value),
        dim_r_(dim_r),
        dim_c_(dim_c)
    {
    }
    ConstantInputNode(const std::string &name, std::shared_ptr<void> tensor_handle, Shape const &tensor_shape) :
        InputNode(name, InputNodeType::Constant, NodeType::kInput),
        node_type_(ConstantInputNodeType::Tensor),
        tensor_handle_(tensor_handle),
        tensor_shape_(tensor_shape),
        dim_r_(-1),
        dim_c_(-1)
    {
        set_shape(tensor_shape);
    }

    virtual std::unique_ptr<Node> clone(std::string const &name = "") const override;
    bool is_single_value() const { return this->node_type_ == ConstantInputNodeType::SingleValue; }
    bool is_tensor() const { return this->node_type_ == ConstantInputNodeType::Tensor; }
    float constant_value() const
    {
        TT_ASSERT(is_single_value());
        return this->constant_value_;
    }
    std::pair<int, int> constant_dims() const
    {
        TT_ASSERT(is_single_value());
        return std::make_pair(dim_r_, dim_c_);
    }
    std::shared_ptr<void> tensor() const
    {
        TT_ASSERT(is_tensor());
        return tensor_handle_;
    }
    void set_tensor_handle(std::shared_ptr<void> t_h)
    {
        TT_ASSERT(is_tensor());
        this->tensor_handle_ = t_h;
    }
    const Shape &tensor_shape() const
    {
        TT_ASSERT(is_tensor());
        return tensor_shape_;
    }
    void set_sparse_forge(sparse::SparseFORGE sparse_forge) { this->sparse_forge = sparse_forge; }
    sparse::SparseFORGE &get_sparse_forge() { return sparse_forge; }

    bool equivalent(const ConstantInputNode *other) const;
};

enum class OutputType
{
    // Internal is used for outputs that are not exposed directly to the user, but used and handled internally for i/o
    // between different graphs, e.g. passing intermediate values from forward graph to the backward graph.
    Internal,

    // Outputs which are defined by/exposed to the user, e.g. result of the forward pass.
    External,
};

class OutputNode : public QueueNode
{
   protected:
    bool requires_grad_;
    bool aliased_tensor_;
    std::string alias_;
    bool is_loss_output_;
    bool is_intermediate_;
    bool untilize_;
    RuntimeTensorTransform runtime_tensor_transform;
    // The golden info is needed if we fractured the output and need to reconstruct it for golden comparison
    std::optional<int> partial_datacopy_golden_output_index;
    std::vector<OpType> golden_transforms;
    OutputType output_type_;

   public:
    OutputNode(std::string name) :
        QueueNode(name, QueueNodeType::Output, NodeType::kOutput),
        requires_grad_(false),
        aliased_tensor_(false),
        is_loss_output_(false),
        is_intermediate_(false),
        untilize_(true),
        output_type_(OutputType::External)
    {
    }
    bool requires_grad() const { return requires_grad_; }
    bool is_loss_output() const { return is_loss_output_; }
    bool is_intermediate() const { return is_intermediate_; }
    bool untilize() const { return untilize_; }
    OutputType output_type() const { return output_type_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    void set_loss_output() { is_loss_output_ = true; }
    void set_intermediate(bool intermediate) { is_intermediate_ = intermediate; }
    void set_untilize(bool should_untilize) { untilize_ = should_untilize; }
    void set_output_type(OutputType output_type) { output_type_ = output_type; }

    void set_alias(const InputNode *node)
    {
        alias_ = node->name();
        aliased_tensor_ = true;
    }

    // Indicates if this output node is actually an alias to an input node. This is used in optimizer graphs, where
    // we want to update a parameter (e.g. `param = param - lr * grad`), but since the rest of the stack doesn't support
    // this yet, we create a new output node that is an alias to the parameter (input) node. So we'll end up with
    // something like this: `updated_param = param - lr * grad`, where `updated_param` is aliased to `param`. Then in
    // the runtime we'll make sure to update the `param` tensor to point to the new data.
    bool is_aliased_tensor() const { return aliased_tensor_; }

    // Returns the name of the input node that this output node is aliased to.
    std::string alias() const { return alias_; }

    virtual std::unique_ptr<Node> clone(std::string const &name = "") const override;

    void set_runtime_tensor_transform(RuntimeTensorTransform transform) { this->runtime_tensor_transform = transform; }

    RuntimeTensorTransform &get_runtime_tensor_transform() { return this->runtime_tensor_transform; }

    void add_golden_transform(OpType const &op_type) { golden_transforms.insert(golden_transforms.begin(), op_type); }
    void set_golden_transforms(std::vector<OpType> const &other) { golden_transforms = other; }
    std::vector<OpType> const &get_golden_transforms() const { return golden_transforms; }
    std::vector<OpType> &get_golden_transforms() { return golden_transforms; }
    void set_partial_datacopy_golden_output_index(int index) { partial_datacopy_golden_output_index = index; }
    std::optional<int> get_partial_datacopy_golden_output_index() { return partial_datacopy_golden_output_index; }
};

struct OpType
{
    friend class ops::Op;
    using Attr = ForgeOpAttr;
    using Attrs = ForgeOpAttrs;

    std::string op_;
    std::vector<Attr> legacy_attrs_;  // legacy path
    Attrs named_attrs_;               // new path

   private:
    ops::Op new_op_;

   public:
    OpType(std::string const &op, std::vector<Attr> const &attr = {}, Attrs named_attrs = {}) :
        op_(op), legacy_attrs_(attr), named_attrs_(std::move(named_attrs)), new_op_(*this)
    {
    }

    bool operator==(const ops::OpType other_type) const { return type() == other_type; }
    bool operator!=(const ops::OpType other_type) const { return !(operator==(other_type)); }

    bool operator==(const OpType &other) const
    {
        return *this == other.type() and legacy_attrs_ == other.legacy_attrs_ and named_attrs_ == other.named_attrs_;
    }
    bool operator!=(const OpType &other) const { return !(*this == other); }

    ops::OpType type() const { return new_op_.type(); }
    ops::Op const &new_op() const { return new_op_; }

    const Attrs &attrs() const { return named_attrs_; }
    bool has_attr(const std::string &attr_name) const { return named_attrs_.find(attr_name) != named_attrs_.end(); }

    template <typename T>
    T const &attr_as(std::string const &name) const
    {
        return std::get<T>(attr(name));
    }

    template <typename T>
    T &attr_as(std::string const &name)
    {
        return std::get<T>(attr(name));
    }

    void set_attr(std::string const &name, Attr attr)
    {
        named_attrs_[name] = attr;
        new_op_.set_attr(name, std::move(attr));
    }

    bool remove_attr(const std::string &attr_name)
    {
        bool removed = named_attrs_.erase(attr_name) > 0;
        if (removed)
            new_op_.remove_attr(attr_name);

        return removed;
    }

    void clear_attrs()
    {
        named_attrs_.clear();
        new_op_.clear_attrs();
    }

    const std::string &name() const { return op_; }

    std::string as_string() const
    {
        std::string ret = op_;
        if (legacy_attrs_.size() > 0)
        {
            ret += "(";
            for (unsigned int i = 0; i < legacy_attrs_.size(); i++)
            {
                if (std::holds_alternative<bool>(legacy_attrs_[i]))
                {
                    ret += std::to_string(std::get<bool>(legacy_attrs_[i])) + ",";
                }
                else if (std::holds_alternative<int>(legacy_attrs_[i]))
                {
                    ret += std::to_string(std::get<int>(legacy_attrs_[i])) + ",";
                }
                else if (std::holds_alternative<float>(legacy_attrs_[i]))
                {
                    ret += std::to_string(std::get<float>(legacy_attrs_[i])) + ",";
                }
                else if (std::holds_alternative<std::string>(legacy_attrs_[i]))
                {
                    ret += std::get<std::string>(legacy_attrs_[i]) + ",";
                }
                else if (std::holds_alternative<std::vector<int>>(legacy_attrs_[i]))
                {
                    auto attr_val = std::get<std::vector<int>>(legacy_attrs_[i]);
                    size_t num_items = attr_val.size();

                    ret += "[";
                    for (size_t j = 0; j < num_items; ++j)
                    {
                        ret += std::to_string(attr_val[j]);
                        if (j < num_items - 1)
                            ret += ", ";
                    }
                    ret += "], ";
                }
                else
                {
                    TT_ASSERT(false, "Unknown alternative in Attr");
                }
            }
            // ret += "(" + std::to_string(attr[0]);
            // for (std::size_t i = 1; i < attr.size(); i++) ret += "," + std::to_string(attr[i]);
            ret += ")";
        }

        if (named_attrs_.size() > 0)
        {
            using tt::operator<<;
            std::stringstream ss;
            bool first = true;
            ss << "{";
            for (auto const &[name, value] : named_attrs_)
            {
                if (not first)
                    ss << ", ";
                ss << name << ": " << value;
                first = false;
            }
            ss << "}";
            ret += ss.str();
        }
        return ret;
    }

    /**
     * Calculations. This is temporary implementation in ops transition period. It will be deleted once all ops are
     * migrated from python to cpp.
     */

    at::Tensor eval(const std::vector<at::Tensor> &tensors) const;
    std::tuple<graphlib::Shape, std::vector<graphlib::DimBroadcast>> shape(
        const std::vector<std::vector<std::uint32_t>> &inputs) const;

    tt::graphlib::NodeContext backward(
        tt::autograd::autograd_context &context,
        int operand,
        const std::vector<tt::graphlib::NodeContext> &inputs,
        const tt::graphlib::NodeContext &output,
        const tt::graphlib::NodeContext &gradient) const;

    template <DecomposeEpoch epoch>
    void decompose(DecomposingContext &dc, const std::vector<tt::graphlib::NodeContext> &inputs) const;

    long initial_flops_estimate(const std::vector<std::vector<std::uint32_t>> &inputs) const;

    bool is_tm() const;
    bool is_eltwise() const;
    bool is_eltwise_unary() const;
    bool is_eltwise_binary() const;
    bool is_eltwise_nary() const;

   private:
    /**
     * Returns attribute based on provided string. Since map::at throws if element does not exist but without any useful
     * info, we will assert to get detailed error message.
     */
    const ForgeOpAttr &attr(const std::string &name) const
    {
        TT_ASSERT(has_attr(name), "Non existing attribute: {}", name);
        return named_attrs_.at(name);
    }

    ForgeOpAttr &attr(const std::string &name)
    {
        TT_ASSERT(has_attr(name), "Non existing attribute: {}", name);
        return named_attrs_.at(name);
    }
};

class OpNode : public TaggedNode
{
   private:
    OpType op_type_;
    bool gradient_op_;  // accumulator op
    std::vector<OpType> golden_transforms;

    // fusing/graph changes have the output of this node be equal to a different golden node
    bool has_golden_id_ = false;
    std::uint32_t golden_id_;

   public:
    OpNode(const std::string &name, const std::string &op_type, NodeType node_type) :
        TaggedNode(name, node_type), op_type_(op_type), gradient_op_(false)
    {
    }
    OpNode(const std::string &name, OpType op_type, NodeType node_type) :
        TaggedNode(name, node_type), op_type_(op_type), gradient_op_(false)
    {
    }

    ops::Op const &new_op() const { return op_type_.new_op(); }
    void change_op_type(OpType const &new_op_type) { op_type_ = new_op_type; }
    void change_op_type(const std::string &new_op_type, std::vector<OpType::Attr> attrs = {})
    {
        op_type_ = OpType(new_op_type, attrs);
    }
    void change_op_type(const std::string &new_op_type, std::vector<OpType::Attr> attrs, OpType::Attrs named_attrs)
    {
        op_type_ = OpType(new_op_type, attrs, named_attrs);
    }
    ops::OpType new_op_type() const { return new_op().type(); }
    OpType const &op_type() const { return op_type_; }
    IRLevel get_ir_level() const { return IRLevel::IR_TT_FORGE; }
    const std::string &op_name() const { return op_type_.name(); }
    const std::vector<OpType::Attr> &op_legacy_attrs() const { return op_type_.legacy_attrs_; }
    const OpType::Attrs &op_named_attrs() { return op_type_.named_attrs_; }
    template <typename T>
    const T &op_attr_as(std::string const &name) const
    {
        return op_type_.attr_as<T>(name);
    }
    void set_op_attr(const std::string &name, OpType::Attr value) { op_type_.set_attr(name, std::move(value)); }

    void set_gradient_op(bool value = true) { gradient_op_ = value; }
    bool is_gradient_op() const { return gradient_op_; }
    bool is_embedding() const
    {
        return new_op_type() == ops::OpType::Embedding || new_op_type() == ops::OpType::EmbeddingBw;
    }
    bool is_matmul() const { return new_op_type() == ops::OpType::Matmul || is_depthwise_matmul(); }
    bool is_reduce() const
    {
        return new_op_type() == ops::OpType::ReduceAvg or new_op_type() == ops::OpType::ReduceMax or
               new_op_type() == ops::OpType::ReduceSum;
    }
    bool is_add() const { return new_op_type() == ops::OpType::Add; }
    bool is_maximum() const { return new_op_type() == ops::OpType::Maximum; }
    bool is_dense_matmul() const { return is_matmul() and not is_sparse_matmul() and not is_depthwise_matmul(); }
    bool is_sparse_matmul() const { return is_matmul() and new_op().has_attr("identity"); }
    // Check whether this is needed, because it makes no sence.
    bool is_depthwise_matmul() const { return new_op_type() == ops::OpType::Depthwise; }
    bool is_matmul_not_sparse() const { return is_matmul() and new_op().has_attr("identity"); }

    bool is_tm() const { return op_type_.is_tm(); };
    bool is_eltwise() const { return op_type_.is_eltwise(); };
    bool is_eltwise_unary() const { return op_type_.is_eltwise_unary(); }
    bool is_eltwise_binary() const { return op_type_.is_eltwise_binary(); }
    bool is_eltwise_nary() const { return op_type_.is_eltwise_nary(); }

    void set_output_df_from_operands(const Graph *graph);
    void add_golden_transform(OpType const &op_type) { golden_transforms.insert(golden_transforms.begin(), op_type); }
    void set_golden_transforms(std::vector<OpType> const &other) { golden_transforms = other; }
    std::vector<OpType> const &get_golden_transforms() const { return golden_transforms; }
    std::vector<OpType> &get_golden_transforms() { return golden_transforms; }

    void set_golden_id(std::uint32_t golden_id)
    {
        has_golden_id_ = true;
        golden_id_ = golden_id;
    }
    void disable_golden_id() { has_golden_id_ = false; }
    bool has_golden_id() const { return has_golden_id_; }
    std::uint32_t golden_id() const
    {
        TT_ASSERT(has_golden_id_);
        return golden_id_;
    }
};

class PyOpNode : public OpNode
{
   public:
    PyOpNode(const std::string &name, const std::string &op_type) : OpNode(name, op_type, NodeType::kPyOp) {}
    PyOpNode(const std::string &name, OpType op_type) : OpNode(name, op_type, NodeType::kPyOp) {}
    virtual std::unique_ptr<Node> clone(std::string const &name = "") const override;

    void copy_parent_op_attributes(PyOpNode *node);
};

// Modifiable edge attributes outside of Edge itself because Edge is mostly immutable in current
// graph design
class EdgeAttributes
{
   private:
    EdgeType edge_type_;
    std::vector<OpType> tms;
    UBlockOrder ublock_order = UBlockOrder::R;

   public:
    EdgeAttributes(EdgeType edge_type) : edge_type_(edge_type) {}
    virtual ~EdgeAttributes() = default;

    bool has_broadcast_dims() const;
    void clear_broadcast_dims();
    void set_broadcast_dim(int dim, int size_or_factor, bool explicit_bcast = false)
    {
        tms.push_back(
            OpType("broadcast", {}, {{"dim", dim}, {"size", size_or_factor}, {"explicit_bcast", explicit_bcast}}));
    }
    void remove_broadcast_dim(int dim);
    inline UBlockOrder get_ublock_order() const { return ublock_order; }
    inline void set_ublock_order(UBlockOrder new_ublock_order) { ublock_order = new_ublock_order; }
    void append_tm(OpType type) { tms.push_back(type); }
    void set_tms(std::vector<OpType> new_tms) { tms = new_tms; }
    void append_tms(std::vector<OpType> new_tms) { tms.insert(tms.end(), new_tms.begin(), new_tms.end()); }
    void prepend_tm(std::string op_type, std::vector<OpType::Attr> attrs)
    {
        tms.insert(tms.begin(), OpType(op_type, attrs));
    }
    void prepend_tm(OpType type) { tms.insert(tms.begin(), type); }

    const std::vector<OpType> &get_tms() const { return tms; }
    std::vector<OpType> &get_tms() { return tms; }

    // Copy values from another edge attributes
    void copy_from(const EdgeAttributes &other)
    {
        tms = other.tms;
        ublock_order = other.ublock_order;
    }

    EdgeType edge_type() const { return edge_type_; }
    bool has_tms() const { return not tms.empty(); }

    bool operator==(EdgeAttributes const &other) const
    {
        return edge_type_ == other.edge_type_ and tms == other.tms and ublock_order == other.ublock_order;
    }

    static std::shared_ptr<EdgeAttributes> create(EdgeType edge_type);

    // Checked casting to sub-node type
    template <typename T>
    static std::shared_ptr<T> as(std::shared_ptr<EdgeAttributes> &base);
    template <typename T>
    static const std::shared_ptr<T> as(const std::shared_ptr<EdgeAttributes> &base);
};

class LoopEdgeAttributes : public EdgeAttributes
{
   public:
    // TypeDefs
    using IterationParametersMap = std::unordered_map<std::string, std::vector<std::string>>;
    struct LoopEdgeAttributesInternal
    {
        int loop_iterations_;
        IterationParametersMap parameter_to_matched_parameters_;
        std::unordered_set<NodeId> nodes_processed_in_loop_;
    };

    explicit LoopEdgeAttributes(EdgeType edge_type) : EdgeAttributes(edge_type) {}
    LoopEdgeAttributes(EdgeType edge_type, const LoopEdgeAttributesInternal &&attributes) :
        EdgeAttributes(edge_type), attributes(std::move(attributes))
    {
    }
    LoopEdgeAttributes(EdgeType edge_type, const LoopEdgeAttributesInternal &attributes) :
        EdgeAttributes(edge_type), attributes(attributes)
    {
    }

    int loop_iterations() const { return this->attributes.loop_iterations_; }
    bool is_processed_in_loop(NodeId node_id) const
    {
        return attributes.nodes_processed_in_loop_.find(node_id) != attributes.nodes_processed_in_loop_.end();
    }
    const std::vector<std::string> matched_parameters(const std::string &parameter) const
    {
        return attributes.parameter_to_matched_parameters_.at(parameter);
    }
    std::string matched_parameters(const std::string &parameter, int loop_iteration_idx) const
    {
        return attributes.parameter_to_matched_parameters_.at(parameter).at(loop_iteration_idx);
    }
    void set_loop_iterations(int loop_iterations) { this->attributes.loop_iterations_ = loop_iterations; }
    void set_iteration_parameters(const IterationParametersMap &parameter_to_matched_parameters)
    {
        this->attributes.parameter_to_matched_parameters_ = parameter_to_matched_parameters;
    }
    void set_nodes_processed_in_loop(const std::unordered_set<NodeId> &nodes_processed_in_loop)
    {
        this->attributes.nodes_processed_in_loop_ = nodes_processed_in_loop;
    }

   private:
    LoopEdgeAttributesInternal attributes;
};

bool op_type_is_accumulate(const std::string &type);

std::ostream &operator<<(std::ostream &out, const NodeType &opcode);
std::ostream &operator<<(std::ostream &out, InputNodeType t);
std::ostream &operator<<(std::ostream &out, const OpType &op_type);
std::ostream &operator<<(std::ostream &out, const UBlockOrder &ublock_order);

inline std::string to_string(InputNodeType t)
{
    switch (t)
    {
        case InputNodeType::Parameter: return "parameter";
        case InputNodeType::Constant: return "constant";
        case InputNodeType::Gradient: return "gradient";
        case InputNodeType::Accumulator: return "accumulator";
        case InputNodeType::Activation: return "activation";
        case InputNodeType::Loss: return "loss";
        case InputNodeType::OptimizerParameter: return "optimizer_parameter";
        case InputNodeType::Target: return "target";
        default: return "unknown";
    }
}

}  // namespace graphlib
}  // namespace tt

template <>
struct fmt::formatter<tt::graphlib::OpType> : fmt::formatter<std::string_view>
{
    inline auto format(const tt::graphlib::OpType &op_type, fmt::format_context &ctx) const -> decltype(ctx.out())
    {
        std::ostringstream oss;
        oss << op_type;
        return fmt::formatter<std::string_view>::format(oss.str(), ctx);
    }
};
