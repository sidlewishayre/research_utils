import os
import sys
import numpy as np
import pandas as pd

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

sys.path.append(CWD)
from reg_runner.regression import Regression
from reg_runner.reg_table import RegTable
from reg_runner.reg_table import RegTables
from other_utils.pandas_utils import records_to_df
from other_utils.arg_utils import is_iterable
from logger import logger
from stata_management.run_stata import get_stata_file


def get_unique_id_names(regs):
    regs = regs.copy()
    regs["repeat_aux"] = 1
    regs["id_repeat_str"] = "_" + regs.groupby("id")["repeat_aux"].cumsum().astype(str)
    regs["repeats"] = regs.groupby("id")["id_repeat_str"].transform("nunique") > 1
    regs.loc[~regs["repeats"], "id_repeat_str"] = ""
    regs["id"] = regs["id"] + regs["id_repeat_str"]
    return regs["id"]


def init_regressions(row):
    return Regression(**row)


def init_table(row):
    return RegTable(**row)


def obj_to_df(obj):
    if type(obj) == dict:
        obj = [obj]
    if is_iterable(obj):
        obj = records_to_df(obj)
        obj = obj.transpose()
    else:
        obj = pd.DataFrame(columns=["id"])
    return obj


def add_save_vars(regs, tables):
    save_vars_cols = ["save_vars", "save_vars_kwargs"]
    save_vars = tables[["id", "extra_info_kwargs"]].dropna().copy()
    for col in save_vars_cols:
        save_vars[col] = save_vars["extra_info_kwargs"].apply(
            lambda x: x[col] if col in x.keys() else None
        )
    save_vars = save_vars.set_index(["id"])[save_vars_cols].dropna()
    for col in save_vars_cols:
        if not col in regs.columns:
            regs[col] = None
    regs[col] = regs["table"].map(save_vars).fillna(regs[col])
    return regs


def run_tables(
    regs,
    save_folder,
    tables=None,
    reload=False,
    overwrite=False,
    prevent_reload=False,
    reg_kwargs={},
    pdf_kwargs={},
    open_pdf=False,
    open_excel=False,
    debug=False,
    run=True,
    reload_save_stats=False,
):
    """
    Pass regs as list of dictionaries or as DataFrame with entries:
    id
    reg_code
    data
    table (optional)
    """
    if debug:
        reg_kwargs["debug"] = True
        save_folder = get_stata_file().replace(".dta", "")
    regs = obj_to_df(regs)
    tables = obj_to_df(tables)
    regs_folder = os.path.join(save_folder, "regs")
    os.makedirs(regs_folder, exist_ok=True)
    if "table" not in regs.columns:
        regs["table"] = None
    if regs["table"].isnull().sum() > 0:
        regs["table"] = regs["table"].fillna("table_1")
        if "table_1" not in tables["id"]:
            tables = pd.concat([tables, pd.DataFrame([["table_1"]], columns=["id"])])
    if regs["id"].value_counts().max() > 1:
        raise ValueError(
            "Repeating regression ids is not allowed. Will prevent future table manipulation."
        )
    if prevent_reload:
        regs["prevent_reload"] = True
    regs["save_folder"] = regs.apply(
        lambda x: os.path.join(regs_folder, x["id"]), axis=1
    )
    if regs["save_folder"].value_counts().max() > 1:
        raise ValueError("Repeating save folder not allowed.")
    if "reload" in regs.columns:
        regs["reload"] = regs["reload"].fillna(False)
        reload_num = regs["reload"].sum()
        if reload_num > 0:
            logger.warning(
                "Reloading {} regs. Are you sure you want to continue?".format(
                    reload_num
                )
            )
    if "prevent_reload" in regs.columns:
        regs["prevent_reload"] = regs["prevent_reload"].fillna(False)
    if reload_save_stats:
        regs["reload_save_stats"] = reload_save_stats
    if "extra_info_kwargs" in tables.columns:
        regs = add_save_vars(regs, tables)
    regs = regs.replace(np.nan, None)
    regs["regression"] = regs.drop(columns="table").apply(init_regressions, axis=1)
    tables = tables.set_index("id")
    tables["regressions"] = regs.groupby("table")["regression"].apply(lambda x: list(x))
    tables = tables.reset_index()
    if tables["regressions"].isnull().sum() > 0:
        len_bef = len(tables)
        tables = tables.dropna(subset=["regressions"])
        logger.warning("Dropping {} tables.".format(len_bef - len(tables)))
    if len(tables) == 0:
        raise ValueError(
            "No tables populated with regressions. Fix regreesions 'table_id'."
        )
    tables = tables.replace(np.nan, None)
    tables["table"] = tables.apply(init_table, axis=1)
    reg_table = RegTables(list(tables["table"]), save_folder)
    if run:
        reg_table.run_tables(reg_kwargs=reg_kwargs, reload=reload, overwrite=overwrite)
        # [reg.results for table in reg_table.tables for reg in table.regressions if reg.results is c None]
        reg_table.gen_tables(
            open_pdf=open_pdf, open_excel=open_excel, pdf_kwargs=pdf_kwargs
        )
    return reg_table
