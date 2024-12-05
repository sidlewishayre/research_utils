import re
import os
import sys
import shutil
import pandas as pd

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

sys.path.append(CWD)
from logger import logger
from file_management.file_manager import FileManager
from stata_management.stata_reg import get_fixed_effects
from stata_management.run_stata import get_regression_results
from stata_management.run_stata import clean_reg_results
from string_formatting import capitalize_words
from other_utils.arg_utils import is_iterable
from other_utils.arg_utils import modify_kwargs
from stata_management.run_stata import make_stata_filepath
from stata_management.run_stata import STATA_COL_LEN
from stata_management.stata_reg import get_reg_dep_var
from stata_management.stata_reg import get_reg_fn
from stata_management.stata_reg import get_sureg_num
from other_utils.arg_utils import compare_gen
from other_utils.arg_expand import expand_to_dict_list
from reg_runner.monitoring_changes import MonitorChanges
from reg_runner.monitoring_changes import compute_batches
from reg_runner.compute_stats import RegStatistic

fm = FileManager()

DEFAULT_SAVE_STATS = ["N", "r2"]

EXTRA_PARAMS = [
    "subset",
    "dep_var",
    "fe",
    "sureg",
    "sureg_num",
    "sureg_keep_cols",
    "reload",
    "prevent_reload",
    "extra_rows",
    "return_change_on_fail",
]

NEW_COMPUTE_PARAMS = [
    "stata_dep_var",
    "stata_command",
    "display_result",
    "save_stats_init",
]


def check_none_str(value):
    return (type(value) == str) and (value.lower() == "none")


def init_param(value, infer_fn, other_fn=None):
    if check_none_str(value):
        return None
    if (value is None) or (not is_iterable(value) and pd.isna(value)):
        return infer_fn()
    if other_fn is not None:
        return other_fn(value)
    return value


class Regression(MonitorChanges):

    def __init__(
        self,
        id,
        reg_code,
        save_folder,
        data,
        stata_code=None,
        dep_var=None,
        subset=None,
        results=None,
        fe=None,
        extra_rows=None,
        sureg=None,
        sureg_num=None,
        sureg_keep_cols=None,
        prevent_reload=False,
        reload=False,
        compute_residuals=None,
        save_residuals_kwargs=None,
        reload_save_stats=None,
        post_estimation_code=None,
        return_change_on_fail=False,
        save_stats=None,
        margins_code=None,
        margins_use=None,
    ):
        self.initialize_monitor_variables(
            id=id,
            save_folder=save_folder,
            results=results,
            prevent_reload=prevent_reload,
            reload=reload,
            obj_filepath_name="reg_class.dmp",
            results_filepath_name="coefs.dta",
            extra_params=EXTRA_PARAMS,
            new_compute_params=NEW_COMPUTE_PARAMS,
            compute_fn=get_regression_results,
            compute_kwargs={"delete_matrix": False},
            compute_kwargs_list={
                "reg_code": "reg_code",
                "matrix_save_filepath": "results_filepath",
                "save_stats": "save_stats_list",
                "save_stats_kwargs": "save_stats_kwargs",
                "save_stats_filepath": "save_stats_filepath",
                "save_residuals_filepath": "residuals_filepath",
                "save_residuals_kwargs": "save_residuals_kwargs",
                "post_estimation_code": "post_estimation_code",
                "margins_code": "margins_code",
                "margins_save_filepath": "margins_save_filepath",
            },
            pool_vars={
                "df": "df",
                "filepath": "filepath",
                "start_stata_code": "stata_code",
            },
            compute_fn_saves=True,
        )
        self.id = id
        self.reg_code = reg_code
        self.save_folder = save_folder
        self.stata_code = stata_code
        self.results = results
        self.data = data
        self.compute_residuals = (
            compute_residuals if compute_residuals is not None else False
        )
        self.save_residuals_kwargs = (
            save_residuals_kwargs if save_residuals_kwargs is not None else {}
        )
        self.df = data if type(data) == pd.DataFrame else None
        self.filepath = data if type(data) == str else None
        if self.df is None and self.filepath is None:
            raise ValueError("Data must be df or filepath.")
        self.fe = init_param(
            value=fe,
            infer_fn=self.get_fixed_effects,
            other_fn=lambda fe: (
                fe
                if type(fe) == dict
                else dict(zip([this_fe for this_fe in fe], ["YES"] * len(fe)))
            ),
        )
        self.extra_rows = init_param(extra_rows, lambda: None)
        self.stata_command = get_reg_fn(self.reg_code)
        self.save_stats_init = save_stats
        self.save_stats_filepath = os.path.join(self.save_folder, "save_stats_info.dta")
        self.reload_save_stats = reload or (
            reload_save_stats is not None and reload_save_stats
        )
        if compute_residuals and "resid_col" not in self.save_residuals_kwargs.keys():
            self.get_default_resid_col()
        self.stata_dep_var = get_reg_dep_var(self.reg_code)
        self.sureg = self.stata_command == "sureg"
        self.sureg_num = sureg_num
        self.sureg_keep_cols = sureg_keep_cols
        self.get_sureg()
        self.dep_var = init_param(dep_var, lambda: self.stata_dep_var)
        self.get_save_stats()
        subset = init_param(subset, self.get_subset)
        repeats = (
            len(self.sureg_keep_cols)
            if self.sureg_keep_cols is not None and self.sureg
            else 1
        )
        self.subset = [subset] * repeats
        self.reload = reload
        self.prevent_reload = prevent_reload
        self.residuals_filepath = (
            self.get_residuals_filepath() if self.compute_residuals else None
        )
        self.margins_code = margins_code
        self.margins_save_filepath = (
            os.path.join(self.save_folder, "margins_coefs.dta")
            if self.margins_code is not None
            else None
        )
        self.margin_use = (
            (
                (margins_use.lower() if not is_iterable(margins_use) else margins_use)
                if margins_use is not None
                else "exclusive"
            )
            if self.margins_code is not None
            else None
        )
        if self.save_residuals_kwargs != {} and not self.compute_residuals:
            raise ValueError(
                "Cannot specify save_residuals_kwargs if not specifying compute_residuals=True."
            )
        self.return_change_on_fail = (
            return_change_on_fail if not pd.isna(return_change_on_fail) else False
        )
        self.no_save_vars = ["save_stats", "save_stats_full", "save_stats_init"]
        self.initialize_save_stats()
        self.initialize_monitor()
        if callable(post_estimation_code):
            self.post_estimation_code = post_estimation_code(**vars(self))
        else:
            self.post_estimation_code = post_estimation_code

    def get_residuals_filepath(self):
        return os.path.join(self.save_folder, "resids.dta")

    def get_default_resid_col(self):
        self.save_residuals_kwargs["resid_col"] = (
            "y_hat"
            if "prediction" in self.save_residuals_kwargs.keys()
            and self.save_residuals_kwargs["prediction"]
            else "resid_col"
        )

    def get_results_exists_extra(self, results_exists, raise_error=False):
        if self.margins_code is not None and not os.path.exists(
            self.margins_save_filepath
        ):
            return False
        if (not self.compute_residuals) or (
            "save_residuals" in self.save_residuals_kwargs.keys()
            and not self.save_residuals_kwargs["save_residuals"]
        ):
            extra_results_exist = results_exists
        else:
            resids_exists = os.path.exists(self.residuals_filepath)
            if resids_exists and not results_exists:
                if raise_error:
                    raise ValueError(
                        "Resids exist but not results for regression {}.".format(
                            self.save_folder
                        )
                    )
                extra_results_exist = False
            elif (results_exists and not resids_exists) and (
                "save_residuals" not in self.save_residuals_kwargs.keys()
                or self.save_residuals_kwargs["save_residuals"]
            ):
                if raise_error:
                    raise ValueError(
                        "Results exist but not results for regression {}. Try reloading regression".format(
                            self.save_folder
                        )
                    )
                extra_results_exist = False
            elif (
                results_exists
                and resids_exists
                and (
                    os.path.getmtime(self.residuals_filepath)
                    < os.path.getmtime(self.results_filepath)
                )
            ):
                raise ValueError("Residuals not updated as recently as results.")
            else:
                extra_results_exist = results_exists
        return extra_results_exist

    def remove_results_extra(self):
        if self.compute_residuals and os.path.exists(self.residuals_filepath):
            os.remove(self.residuals_filepath)

    def get_sureg(self):
        if self.sureg:
            self.stata_dep_var = [
                var
                for i, var in enumerate(self.stata_dep_var)
                if i in self.sureg_keep_cols
            ]
        if self.sureg_num is None and self.sureg:
            self.sureg_num = get_sureg_num(self.reg_code)
        if self.sureg_keep_cols is None and self.sureg_num is not None:
            self.sureg_keep_cols = range(self.sureg_num)

    def get_save_stats(self):
        if self.save_stats_init is None:
            if not self.sureg:
                save_stats = DEFAULT_SAVE_STATS
            else:
                save_stats = ["N"] + [
                    "r2_{}".format(i + 1) for i in range(self.sureg_num)
                ]
        elif check_none_str(self.save_stats_init):
            save_stats = None
        elif type(self.save_stats_init) == list:
            save_stats = self.save_stats_init
        save_stats = expand_to_dict_list(save_stats, var="id")
        self.save_stats = [
            RegStatistic(
                reg=self,
                reload=self.reload_save_stats,
                prevent_reload=self.prevent_reload,
                **stat,
            )
            for stat in save_stats
        ]
        for stat in self.save_stats:
            stat.initialize()
        if self.reload_save_stats:
            self.compute_save_stats(reload=True, overwrite=True, raise_error=False)
        self.save_stats_full = self.save_stats
        for stat in self.save_stats:
            self.save_stats_full = [
                var
                for var in list(stat.dependent_stats.values())
                if var.id
                not in [stat.id for stat in self.save_stats_full]
                + [stat.id for stat in self.save_stats]
            ] + self.save_stats_full
        self.get_save_stat_kwargs()
        id_list = [
            stat.id if stat.reg_type != "stata" else stat.id[:STATA_COL_LEN]
            for stat in self.save_stats
        ]
        if len(pd.Series(id_list).drop_duplicates()) != len(id_list):
            raise ValueError("Cannot repeat save_stats ids.")

    def get_save_stat_kwargs(self):
        stats_list = self.save_stats_full
        self.save_stats_list = [
            stat.id for stat in stats_list if stat.reg_type == "stata"
        ]
        self.save_stats_kwargs = [
            {"stata_code": stat.stata_code, "save_var": stat.stata_save_var}
            for stat in stats_list
            if stat.reg_type == "stata"
        ]

    def is_save_var_in_results(self, k):
        return k in self.results.index and (~self.results.loc[k].isnull()).sum() > 0

    def check_save_stats(
        self, check_fn=lambda stat: stat.check_complete(), raise_error=True
    ):
        for stat in self.save_stats:
            if not check_fn(stat):
                return False
        return True

    def check_complete_extra(self, complete):
        if not complete:
            return False
        save_stats_bool = self.check_save_stats()
        if not save_stats_bool:
            if not self.reload and self.prevent_reload:
                success = self.compute_save_stats(raise_error=False)
                if success:
                    return True
            return False
        return True

    def initialize_save_stats(self):
        for stat in self.save_stats:
            stat.initialize_monitor()

    def get_display_results(self):
        if self.extra_rows is None:
            self.display_results = self.results
        elif type(self.extra_rows) != dict:
            raise ValueError("'extra_rows' must be dict with row name and values.")
        elif self.results.columns.isin(self.extra_rows.keys()).any():
            raise ValueError(
                "In Regression {}, 'extra_rows' cannot be in regression results.".format(
                    self.id
                )
            )
        else:
            self.extra_rows = {
                k: v if is_iterable(v) and len(v) == 2 else [v, ""]
                for k, v in self.extra_rows.items()
            }
            self.display_results = pd.concat(
                [self.results, pd.DataFrame(self.extra_rows, index=["b", "se"])], axis=1
            )
            self.display_results.index.name = self.results.index.name
        return self.display_results

    def compute_save_stats(self, reload=False, overwrite=False, raise_error=True):
        compute_batches(
            self.save_stats,
            reload=reload or self.reload_save_stats,
            overwrite=overwrite,
            kwargs={"raise_error": raise_error},
        )
        return self.check_save_stats()

    def get_subset(self):
        match = re.search(" if ", self.reg_code)
        if match is None:
            return None
        else:
            subset = self.reg_code[match.span()[1] :].split(",")[0]
            subset = subset.replace("1 == 1", "")
            subset = subset.replace("== 1", "")
            subset = subset.replace("&", ",")
            subset = subset.replace(" ,", ",")
            subset = re.sub("[ ]+", " ", subset)
            subset = re.sub("^[ ,]+", "", subset)
            subset = re.sub("[ ,]+$", "", subset)
            if re.match("^[A-Za-z\- ]+$", subset) is not None:
                subset = capitalize_words(subset)
            return subset

    def clean_reg_results(self, results):
        return clean_reg_results(results)

    def load_raw_results_extra(self, raise_error=True):
        self.results = self.clean_reg_results(self.results)
        if self.margins_code is not None and self.margin_use != "none":
            if not os.path.exists(self.margins_save_filepath):
                if raise_error:
                    raise ValueError(
                        "margins_save_filepath does not exist for reg {}".format(
                            self.id
                        )
                    )
                return False
            margins_results = clean_reg_results(fm.load(self.margins_save_filepath))
            marings_results = margins_results.reindex(self.results.index)
            marings_results = margins_results.reindex(self.results.columns, axis=1)
            if self.margin_use == "prioritize":
                self.results = margins_results.fillna(self.results)
            elif is_iterable(self.margin_use):
                for row in self.margin_use:
                    self.results.loc[row] = margins_results.loc[row]
        return True

    def run_reg(self, extra_objs, overwrite=False, kwargs={}):
        self.compute(extra_objs=extra_objs, overwrite=overwrite, kwargs=kwargs)

    def compute_extra(self, obj_list):
        for reg in obj_list:
            reg.compute_save_stats(overwrite=True)
            reg.save()

    def get_fixed_effects(self, extra_regressions=None):
        reg_list = self.get_pool_list(extra_regressions, extra_pool_vars=["var_map"])
        reg_code_list = [reg.reg_code for reg in reg_list]
        if len(reg_code_list) == 1:
            reg_code_list = reg_code_list[0]
        return get_fixed_effects(reg_code_list, var_map=None)

    def get_load_resid_code(self):
        stata_code = f"""
        merge 1:1 {" ".join(self.save_residuals_kwargs['id_cols'])} using {make_stata_filepath(self.residuals_filepath)}
        drop if _merge == 2
        rename {self.save_residuals_kwargs['resid_col']} {self.id[:STATA_COL_LEN]}
        drop _merge
        """
        return stata_code

    def load_residuals(
        self, extra_regressions=[], extra_stata_code=None, **load_stata_kwargs
    ):
        if not self.compute_residuals:
            raise ValueError(
                "Must specify save_residuals to be True and run regression."
            )
        if not os.path.exists(self.residuals_filepath):
            raise ValueError("Must call run_reg first.")
        if "id_cols" not in self.save_residuals_kwargs.keys():
            raise ValueError("How is this possible? Must specify id_cols.")
        stata_code = self.get_load_resid_code()
        for reg in extra_regressions:
            stata_code = stata_code + "\n" + reg.get_load_resid_code()
        if extra_stata_code is not None:
            stata_code = stata_code + "\n" + extra_stata_code + "\n"
        return fm.load_stata(
            filepath=self.data, stata_code=stata_code, **load_stata_kwargs
        )
