# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment, PatternFill, Font, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation


# Intialization
all_out_dict = {}
failure_reason_lines = []
total_count = 0
fail_count = 0
skip_count = 0
pass_count = 0


def write_excelfile(content, ws):
    """
    Function is used to write excel sheet
    ----------
    Parameters
    ----------
    content : Data dictionary
    ws      : Need to write data on this worksheet
    """
    fields = ["JOB_NAME", "STATUS"]
    ws.append(fields)
    for jb_name, info in content.items():
        ws.append([jb_name] + [info])
    align_excel(ws)


def align_excel(content_sheet):
    """
    Function is used to align by adding border, adjusting font size, adjusting spaces and coloring the data validation choices
    ----------
    Parameter
    ----------
    content_sheet   : Sheet needs to be aligned
    """
    working_sheet = content_sheet

    # Adding Border
    side = Side(style="thin", color="000000")
    thin_border = Border(left=side, right=side, top=side, bottom=side)
    for row in working_sheet.iter_rows(
        min_row=1, max_row=working_sheet.max_row, min_col=1, max_col=working_sheet.max_column
    ):
        for cell in row:
            cell.border = thin_border

    # Header Alignment
    blue_fill = PatternFill(start_color="d4e2f8", end_color="d4e2f8", fill_type="solid")
    center_aligned = Alignment(horizontal="center", vertical="center")
    for col in range(1, 3):
        working_sheet.cell(row=1, column=col).fill = blue_fill
        working_sheet.cell(row=1, column=col).font = Font(bold=True)
        working_sheet.cell(row=1, column=col).alignment = center_aligned

    # Calculating and Assigning column width
    for col_num in range(1, working_sheet.max_column + 1):
        max_length = 0

        for row_num in range(1, working_sheet.max_row + 1):
            cell = working_sheet.cell(row=row_num, column=col_num)

            # Calculate the length of the cell's content (if it's not None)
            if cell.value:
                cell_length = len(str(cell.value))
                max_length = max(max_length, cell_length)

        column_letter = get_column_letter(col_num)
        working_sheet.column_dimensions[column_letter].width = max_length + 2  # adding column offset ( 2)

    if working_sheet.title == "Failure_Reason":
        for row_index in range(2, working_sheet.max_row + 1):
            working_sheet.cell(row=row_index, column=1).alignment = Alignment(horizontal="left", vertical="center")

            # Calculating and Assigning row height
            line_count = 0
            cell = working_sheet.cell(row=row_index, column=2)
            list_value = str(cell.value).split("\n")
            for line in list_value:
                if line:
                    line_count = line_count + 1

            column_letter = get_column_letter(col_num)
            working_sheet.row_dimensions[row_index].height = 20 * line_count

    elif working_sheet.title == "All_Runner_Results":
        choices = ["PASS", "FAIL", "SKIP"]
        dv = DataValidation(type="list", formula1=f'"{",".join(choices)}"', showDropDown=True)
        working_sheet.add_data_validation(dv)

        # Fixing color for DataValidation choices
        pass_fill = PatternFill(start_color="afd8bc", end_color="afd8bc", fill_type="solid")  # light green
        fail_fill = PatternFill(start_color="cc0000", end_color="cc0000", fill_type="solid")  # light Red
        skip_fill = PatternFill(start_color="ffe599", end_color="ffe599", fill_type="solid")  # light yello
        for row_index in range(2, working_sheet.max_row + 1):
            working_sheet.cell(row=row_index, column=1).alignment = Alignment(horizontal="left", vertical="center")
            cell = working_sheet[f"B{row_index}"]
            dv.add(cell)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if cell.value == "PASS":
                cell.fill = pass_fill
            elif cell.value == "FAIL":
                cell.fill = fail_fill
            elif cell.value == "SKIP":
                cell.fill = skip_fill


def add_testcase(test_name, value):
    """
    Adding data in dictionary
    ----------
    Parameters
    ----------
    test_name   : passing line in which testcase name is presented
    value       : status of the testcase ('PASS','FAIL','SKIP')
    """
    extract_testname = (test_name.split("py::"))[1]  # extracting only variant name
    all_out_dict[extract_testname] = value
    test_log_status = "ended"
    return test_log_status


def get_failure_details(failure_reason_lines):
    """
    Function is used to segregate failure details from entire failure log and map it to their respective test cases
    ----------
    Parameter
    ----------
    failure_reason_lines    : List of failure lines
    """
    test_pattern = r"^_+\s+test_[a-zA-Z0-9_]+(\[.*?\])?\s+_+$"
    out_dict = {}
    last_test_line = ""
    before_line_1 = ""
    before_line_2 = ""
    after_line_1 = ""
    after_line_2 = ""
    extra_error_lines = 3
    extract_error = []
    index_cnt = 0
    test_logs = False
    for line in failure_reason_lines:
        if re.match(test_pattern, line):
            test_logs = True
            if line in out_dict:
                print("Readding same test skipped")
            else:
                last_test_line = line

        if line.startswith("E  ") and test_logs == True:

            for i in range(-extra_error_lines, (extra_error_lines + 1)):
                extract_error.append(failure_reason_lines[index_cnt - i])

            key = last_test_line.strip("_ ")
            out_dict[key] = "\n".join(extract_error) + "\n"
            extract_error = []
            test_logs = False

        index_cnt = index_cnt + 1

    write_excelfile(out_dict, new_ws_fr)


if __name__ == "__main__":
    runner_count = 4
    log_path_list = []
    for i in range(1, (runner_count + 1)):
        runner_path = "Logs/pytest_runner" + str(i) + ".log"
        log_path_list.append(runner_path)

    new_wb = Workbook()
    initial_ws1 = new_wb.active
    new_wb.remove(initial_ws1)
    new_ws = new_wb.create_sheet(title="All_Runner_Results")
    new_ws_fr = new_wb.create_sheet(title="Failure_Reason")

    for log_file in log_path_list:
        with open(log_file, "r") as fp:
            lines = fp.readlines()

        extract_testname = ""
        consolidate_data = []
        append_lines = False
        consolidate_result = False
        iteration = False
        test_log_status = "ended"

        for line in lines:
            if "==== short test summary info ====" in line:
                consolidate_result = True

            if consolidate_result == True:
                consolidate_data.append(line)
                continue

            if "warnings summary" in line:
                continue

            if "plugins: " in line:
                iteration = True
            elif iteration == False:
                continue

            if ("forge/test/" in line) and (iteration == True):
                test_case_name = line.strip("\n").split(" ")  # (" "))#("] "))
                test_log_status = "started"

            if ("PASSED" in line) and (test_log_status == "started"):
                test_log_status = add_testcase(test_case_name[0], "PASS")
                total_count = total_count + 1
                pass_count = pass_count + 1

            if ("FAILED" in line) and (test_log_status == "started"):
                test_log_status = add_testcase(test_case_name[0], "FAIL")
                total_count = total_count + 1
                fail_count = fail_count + 1

            if (" SKIPPED" in line) and (test_log_status == "started"):
                extract_testname = ((line.split("py::"))[1]).split(" SKIPPED")
                all_out_dict[extract_testname[0]] = "SKIP"
                test_log_status = "ended"
                total_count = total_count + 1
                skip_count = skip_count + 1

            if "=== FAILURES ====" in line:
                append_lines = True

            if append_lines:
                failure_reason_lines.append(line.strip("\n"))

            if "warnings summary" in line:
                break

    print(
        "\n==============================" + "\033[1m",
        "Short Test Summary",
        "\033[0m" + "=============================",
    )
    print("TOTAL COUNT :", total_count)
    print("PASS COUNT  :", pass_count)
    print("FAIL COUNT  :", fail_count)
    print("SKIP COUNT  :", skip_count)
    print("================================================================================")
    write_excelfile(all_out_dict, new_ws)
    get_failure_details(failure_reason_lines)

    # Saving the workbook
    new_wb.save("Nightly_workbook.xlsx")
    print("\nNightly_workbook.xlsx was created\n")
