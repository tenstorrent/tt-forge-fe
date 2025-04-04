# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from loguru import logger
import pandas as pd
from tabulate import tabulate
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from utils import CompilerComponent
from collections import defaultdict


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
        self.tab = "\t"

    def write(self, data: str):
        self.file.write(data)

    def write_line(self, data: str):
        self.write(data + "\n")

    def write_html_heading(self, heading: str, heading_rank: int = 1):
        heading = f"<h{heading_rank}>{heading}</h{heading_rank}>"
        self.write_line(heading)

    def write_html_paragraph(self, content: str):
        self.write_line(f"<p>{content}</p>")

    @classmethod
    def create_html_bold_text(cls, text: str):
        return f"<b>{text}</b>"

    @classmethod
    def create_html_paragraph(cls, text: str):
        return f"<p>{text}</p>"

    @classmethod
    def create_html_link(cls, link_text: str, url_or_path: str):
        return f'<a href="{url_or_path}">{link_text}</a>'

    @classmethod
    def create_html_list(cls, items: List[str]):
        """
        Generates an HTML unordered list from a list of strings.

        Args:
            items (List[str]): A list of strings to include as list items in the HTML.

        Returns:
            str: The generated HTML unordered list as a string.
        """
        list_items = [f"<li>{item}</li>" for item in items]
        html_list = "<ul>" + "".join(list_items) + "</ul>"
        return html_list

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

    def write_html_table_description(self, table_description: str):
        self.write_html_paragraph(content=table_description)

    def write_html_table_column_description(self, table_column_description: Dict[str, str]):
        """
        Write an HTML-formatted description of table columns to the markdown file.

        This method takes a dictionary of column descriptions, formats each column name
        as bold text followed by its description, and writes the content as an HTML list.

        Args:
            table_column_description (Dict[str, str]): A dictionary where the keys are column names
                and the values are their respective descriptions.
        """
        html_table_column_description = []
        for column_name, column_description in table_column_description.items():
            html_table_column_description.append(
                MarkDownWriter.create_html_bold_text(column_name + ": ") + column_description
            )
        html_table_column_description = MarkDownWriter.create_html_list(html_table_column_description)
        self.write_line(html_table_column_description)

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

        table_df.index = pd.RangeIndex(start=1, stop=int(len(table_df) + 1), step=1)

        html_table = table_df.to_html(index=True, na_rep=" ", justify="center", escape=False)

        self.write_line(html_table)

    def generate_html_table_header(self, headers: Union[List[str], Dict[str, Any]], text_align: str):
        """
        Generates the HTML table header with multi-level support.

        Args:
            headers (Union[List[str], Dict[str, Any]]): The structure of the table headers, which can be a nested dictionary or a simple list.
            text_align (str): The text alignment style for the headers (e.g., 'left', 'center', 'right').

        Returns:
            str: The generated HTML string for the table header.
        """

        def calculate_colspan(header):
            """Recursively calculate the colspan for a header."""
            if isinstance(header, dict):
                return sum(calculate_colspan(value) for value in header.values())
            elif isinstance(header, list):
                return len(header)
            return 1

        def generate_rows(headers, current_level=0, rows=None):
            """Generate all rows for the HTML table header."""
            if rows is None:
                rows = []
            if len(rows) <= current_level:
                rows.append([])

            for key, value in headers.items():
                colspan = calculate_colspan(value)
                rows[current_level].append(f'<th colspan="{colspan}">{key}</th>\n')
                if isinstance(value, dict):
                    generate_rows(value, current_level + 1, rows)
                elif isinstance(value, list):
                    if len(rows) <= current_level + 1:
                        rows.append([])
                    for header in value:
                        rows[current_level + 1].append(f"<th>{header}</th>\n")

            return rows

        # Initialize the HTML table header
        html = f"{self.tab}<thead>\n"

        if isinstance(headers, dict):
            # Generate rows dynamically based on input structure
            rows = generate_rows(headers)
        elif isinstance(headers, list):
            rows = [f"<th>{header}</th>\n" for header in headers]
            rows = [rows]

        # Add rows to the HTML table header
        for row in rows:
            html += f'{self.tab*2}<tr style="text-align: {text_align};">\n{self.tab*3}'
            html += f"{self.tab*3}".join(row)
            html += f"{self.tab*2}</tr>\n"

        # Close the table header
        html += f"{self.tab}</thead>\n"

        return html

    def generate_html_table_body(self, rows: List[List[Any]], rowspan_columns: Optional[List[int]] = None):
        """
        Generates the HTML table body with optional rowspan support.

        Args:
            rows (List[List[Any]]): The table rows, where each row is a list of cell values.
            rowspan_columns (Optional[List[int]]): List of column indices to apply rowspan.

        Returns:
            str: The generated HTML string for the table body.
        """

        def calculate_rowspans(rows, rowspan_columns):
            """Calculate rowspan for each cell, requiring all previous columns to match."""
            rowspan_info = defaultdict(lambda: defaultdict(int))

            for col in rowspan_columns:
                prev_row = None
                prev_index = None
                count = 0

                for i, row in enumerate(rows):
                    # Ensure the row has enough columns
                    if col >= len(row):
                        continue

                    # Check if all columns up to `col` match with the previous row
                    if prev_row is not None and all(row[k] == prev_row[k] for k in range(col + 1)):
                        count += 1
                        rowspan_info[prev_index][col] = count
                        rowspan_info[i][col] = 0  # Hide this cell
                    else:
                        count = 1
                        rowspan_info[i][col] = 1
                        prev_row = row
                        prev_index = i

            return rowspan_info

        # Default to an empty list if rowspan_columns is not provided
        if rowspan_columns is None:
            rowspan_columns = []

        # Calculate rowspans
        rowspan_info = calculate_rowspans(rows, rowspan_columns)

        # Build the HTML table body
        html = f"{self.tab}<tbody>\n"

        for i, row in enumerate(rows):
            html += f"{self.tab * 2}<tr>\n"
            for j, cell in enumerate(row):
                if j in rowspan_columns and rowspan_info[i][j] > 0:
                    html += f'{self.tab * 3}<td rowspan="{rowspan_info[i][j]}">{cell}</td>\n'
                elif j not in rowspan_columns or rowspan_info[i][j] == 1:
                    html += f"{self.tab * 3}<td>{cell}</td>\n"
            html += f"{self.tab * 2}</tr>\n"

        html += f"{self.tab}</tbody>\n"
        return html

    def write_html_table(
        self,
        table_headers: Union[List[str], Dict[str, Any]],
        table_rows: List[List[Any]],
        border: int = 2,
        header_text_align: str = "center",
        rowspan_columns: Optional[List[int]] = None,
    ):
        """
        Generates HTML table based upon the table_headers and table_rows provided

        Args:
            table_headers (Union[List[str], Dict[str, Any]]): The table headers structure.
            table_rows (List[List[Any]]): The table rows.
            border (int): The border width for the table.
            header_text_align (str): The text alignment for the headers.
            rowspan_columns (Optional[List[int]]): Columns to apply rowspan.
        """
        html_table_header = self.generate_html_table_header(table_headers, header_text_align)
        html_table_body = self.generate_html_table_body(table_rows, rowspan_columns)
        html_table = f'<table border="{border}">\n'
        html_table += html_table_header
        html_table += html_table_body
        html_table += "</table>"
        self.write_line(html_table)

    def close_file(self):
        self.file.close()
