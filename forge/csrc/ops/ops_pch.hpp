// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Precompiled header for ops library.
 */
#pragma once

// Standard library headers.
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

// Torch headers.
#include <torch/extension.h>
#include <torch/torch.h>

// Project headers.
#include "autograd/autograd.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "op.hpp"
#include "op_interface.hpp"
#include "passes/decomposing_context.hpp"
#include "utils/assert.hpp"
