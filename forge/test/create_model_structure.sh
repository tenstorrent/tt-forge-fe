# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# !/bin/bash

# : << 'COMMENT'
# SAMPLE:
# Script run command : bash forge/test/create_model_structure.sh

# INPUTS:
# Enter model name : resnet
# Enter model category [text/vision/audio/timeseries/multimodal] : vision
# Enter framework of the model [onnx/pytorch/tensorflow/tflite] : pytorch
# Would you like to create a utils package inside the resnet? [Y/N] : Y
# Would you like to create a tests package for block or Ops tests inside the resnet? [Y/N] : Y
# Would you like to create blocks package inside the tests package? [Y/N] : Y
# Would you like to create ops package inside the tests package? [Y/N] : Y

# OUTPUT:
# tt-forge-fe/forge/test/models/
# |-- __init__.py
# |
# `-- pytorch
#     |-- __init__.py
#     `-- vision
#         |-- __init__.py
#         `-- resnet
#             |-- __init__.py
#             |-- tests
#             |   |-- __init__.py
#             |   |-- blocks
#             |   |   `-- __init__.py
#             |   `-- ops
#             |       `-- __init__.py
#             `-- utils
#                 `-- __init__.py
#
# COMMENT


root_directory_path="${PWD}/forge/test/models"

create_dir_and_file() {
    local path="$1"

    if [[ "$path" == */ ]]; then
        # If the path ends with a slash, treat it as a directory
        if [ ! -d "$path" ]; then
            mkdir -p "$path"
            if [ $? -ne 0 ]; then
                echo "Error: Failed to create directory $path" >&2
                return 1
            fi
            echo "Directory created: $path"
        else
            echo "Directory already exists: $path"
        fi
    else
        # Otherwise, treat it as a file
        local directory=$(dirname "$path")
        if [ ! -d "$directory" ]; then
            mkdir -p "$directory"
            if [ $? -ne 0 ]; then
                echo "Error: Failed to create parent directory for file $path" >&2
                return 1
            fi
            echo "Parent directory created: $directory"
        fi

        if [ ! -f "$path" ]; then
            touch "$path"
            if [ $? -ne 0 ]; then
                echo "Error: Failed to create file $path" >&2
                return 1
            fi
            echo "File created: $path"
        else
            echo "File already exists: $path"
        fi
    fi
}


# Getting inputs from the user
get_inputs() {
    read -p "Enter model name : " model_name
    read -p "Enter model category [text/vision/audio/timeseries/multimodal] : " category
    read -p "Enter framework of the model [onnx/pytorch/tensorflow/tflite] : " framework_name

    if [ -z "$model_name" -o -z "$category" -o -z "$framework_name" ]; then
        echo "Error: model name, model category and framework of the model must be provided."
        exit 1
    fi

    if [ "$category" != "text" -a "$category" != "vision" -a "$category" != "audio" -a "$category" != "timeseries" -a "$category" != "multimodal" ]; then
        echo "Error: model category should be text or vision or audio or timeseries or multimodal but you have provided $category."
        exit 1
    fi

    if [ "$framework_name" != "onnx" -a "$framework_name" != "pytorch" -a "$framework_name" != "tensorflow" -a "$framework_name" != "tflite" ]; then
        echo "Error: framework $framework_name sepecified for the model $model_name is not valid."
        exit 1
    fi

    read -p "Would you like to create a utils package inside the $model_name? [Y/N] : " has_utils
    if [ "$has_utils" != "Y" -a "$has_utils" != "N" ] ; then
        echo "Error: You should provided either Y or N but you have entered $has_utils"
        exit 1
    fi

    read -p "Would you like to create a tests package for block or Ops tests inside the $model_name? [Y/N] : " has_tests
    if [ "$has_tests" = "Y" ] ; then
        read -p "Would you like to create blocks package inside the tests package? [Y/N] : " has_blocks
        read -p "Would you like to create ops package inside the tests package? [Y/N] : " has_ops
    else
        has_tests="N"
        has_blocks="N"
        has_ops="N"
    fi

    echo "$model_name,$category,$framework_name,$has_utils,$has_tests,$has_blocks,$has_ops"
}

inputs=$(get_inputs)
if [[ "$inputs" == "Error:"* ]]; then
    echo $inputs
    exit 1
fi

IFS=',' read -r model_name category framework_name has_utils has_tests has_blocks has_ops <<< "$inputs"

#create a parent and model directory
create_dir_and_file "$root_directory_path/__init__.py"
create_dir_and_file "$root_directory_path/$framework_name/__init__.py"
create_dir_and_file "$root_directory_path/$framework_name/$category/__init__.py"
create_dir_and_file "$root_directory_path/$framework_name/$category/$model_name/__init__.py"


# Create a tests structure
if [ "$has_tests" = "Y" ]; then

    create_dir_and_file "$root_directory_path/$framework_name/$category/$model_name/tests/__init__.py"

    if [ "$has_blocks" = "Y" ]; then
        create_dir_and_file "$root_directory_path/$framework_name/$category/$model_name/tests/blocks/__init__.py"
    fi

    if [ "$has_ops" = "Y" ]; then
        create_dir_and_file "$root_directory_path/$framework_name/$category/$model_name/tests/ops/__init__.py"
    fi

fi

# Create a utils structure
if [ "$has_utils" = "Y" ]; then
    create_dir_and_file "$root_directory_path/$framework_name/$category/$model_name/utils/__init__.py"
fi

tree $root_directory_path/$framework_name/
