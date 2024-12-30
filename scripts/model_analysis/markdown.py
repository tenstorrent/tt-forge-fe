# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pandas as pd
from tabulate import tabulate
from enum import Enum
from typing import Dict, List
from utils import CompilerComponent


class HtmlSymbol(Enum):
    PASS = "&#x2705;"  # Checkmark
    FAIL = "&#x274C;"  # Crossmark
    UNKNOWN = "&#xFFFD;"  # Question mark


class MarkDownWriter:
    """
    A utility class for writing Markdown files, including headings, tables, and links.

    Attributes:
        markdown_file_name (str): The name of the Markdown file (without extension).
        markdown_file_dir_path (str): The directory path where the Markdown file is created.
    """

    def __init__(self, markdown_file_name: str, markdown_file_dir_path: str = None, open_file: bool = True):
        self.markdown_file_name = markdown_file_name
        self.markdown_file = self.markdown_file_name + ".md"
        if markdown_file_dir_path is not None:
            self.markdown_file_dir_path = markdown_file_dir_path
        else:
            self.markdown_file_dir_path = os.getcwd()
        os.makedirs(self.markdown_file_dir_path, exist_ok=True)
        if open_file:
            self.file = open(os.path.join(self.markdown_file_dir_path, self.markdown_file), "w")

    def write(self, data: str):
        self.file.write(data)

    def write_line(self, data: str):
        self.write(data + "\n")

    def write_table_heading(self, table_heading: str, heading_rank: int = 1):
        table_heading = str("#" * heading_rank) + " " + table_heading
        self.write_line(table_heading)

    def write_table(self, headers, rows):
        # Create a Markdown table using the tabulate library with GitHub-flavored table formatting.
        markdown_table = tabulate(rows, headers, tablefmt="github", colalign=("center",) * len(headers))
        self.write_line(markdown_table)

    @classmethod
    def get_component_names_for_header(cls, compiler_component: CompilerComponent):
        if compiler_component == CompilerComponent.FORGE:
            return "Forge-Fe"
        elif compiler_component == CompilerComponent.MLIR:
            return "MLIR"
        elif compiler_component == CompilerComponent.TT_METAL:
            return "Metalium"
        elif compiler_component == CompilerComponent.UNKNOWN:
            return "N/A"
        else:
            logger.error(f"There is no compilercomponent {compiler_component.name}")

    def write_html_table_heading(self, table_heading: str, heading_rank: int = 1):
        table_heading = f"<h{heading_rank}>{table_heading}</h{heading_rank}>"
        self.write_line(table_heading)

    def create_html_table_and_write(self, headers: Dict[str, List[str]], rows: List[List[str]]):
        sub_headers = []
        for headers_list in headers.values():
            sub_headers.extend(headers_list)

        sub_header_row_data_length_match = all([True if len(row) == len(sub_headers) else False for row in rows])

        assert sub_header_row_data_length_match, "Sub headers and table row length is not matched"

        table_df = pd.DataFrame(data=rows, columns=sub_headers)

        top_headers = [
            (main_header, sub_header) for main_header, sub_headers in headers.items() for sub_header in sub_headers
        ]
        table_df.columns = pd.MultiIndex.from_tuples(top_headers)

        html_table = table_df.to_html(index=False, na_rep=" ", justify="center", escape=False)

        self.write_line(html_table)

    @classmethod
    def create_md_link(cls, link_text: str, url_or_path: str):
        return f"[{link_text}]({url_or_path})"

    @classmethod
    def create_html_link(cls, link_text: str, url_or_path: str):
        return f'<a href="{url_or_path}">{link_text}</a>'

    def close_file(self):
        self.file.close()
